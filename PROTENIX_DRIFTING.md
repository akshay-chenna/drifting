# Drifting for Protenix: replacing or augmenting the diffusion module

**Status:** design document. No code. The purpose is to say precisely what would have to be built, what each piece
would cost, which parts are load-bearing, and — most importantly — which of them is likely to fail and how you would
know early.

**Provenance of the code facts below.** Module paths, class names and defaults were read from the Protenix checkout
at `/paperspace/apps/Protenix` (the Protenix-v1 release tree). Protenix v2 keeps the same module boundaries — a
Pairformer trunk feeding a conditioned diffusion module, a mini-rollout for confidence and permutation — so the
integration points are the same, but **verify every symbol against your v2 checkout before relying on it**. Where a
number below is a config default it is marked as such.

**Background required.** `THEORY.md` §§2–6 (the field and its equilibrium), §11 (SE(3)), §12 (force guidance),
§13 (latent and learned priors); Notebooks 3, 7, 8, 9. The two results this document leans on hardest are Notebook 8
§4 (the single-positive fixed point is the reference structure) and Notebook 9 §6 (weight-space guidance is safe in
constrained output spaces, direction-space guidance is not).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What Protenix's diffusion module actually does](#2-what-protenixs-diffusion-module-actually-does)
3. [Why drifting is a candidate at all](#3-why-drifting-is-a-candidate-at-all)
4. [The five risks, stated before the design](#4-the-five-risks-stated-before-the-design)
5. [Three architectures](#5-three-architectures)
6. [Design A in detail: the one-step drifting head](#6-design-a-in-detail-the-one-step-drifting-head)
7. [The field, concretely, for structures](#7-the-field-concretely-for-structures)
8. [Where diversity comes from](#8-where-diversity-comes-from)
9. [Physical guidance from energies and forces](#9-physical-guidance-from-energies-and-forces)
10. [Design C: drifting for the conformational ensemble](#10-design-c-drifting-for-the-conformational-ensemble)
11. [Training plan, staging, and compute](#11-training-plan-staging-and-compute)
12. [Evaluation protocol](#12-evaluation-protocol)
13. [Risk table and kill criteria](#13-risk-table-and-kill-criteria)
14. [Open questions](#14-open-questions)
15. [Empirical update: what picodrift measured](#15-empirical-update-what-picodrift-measured)

---

## 1. Executive summary

**The proposal.** Add a drifting head that maps trunk conditioning plus noise directly to coordinates in **one
network evaluation**, replacing the 200-step diffusion sampler at inference. Train it from scratch against the
existing dataset, using an SE(3)-equivariant drifting field with an augmented target set (Drift-React's
construction), on top of the *frozen* Protenix trunk.

**What it would buy.**

| | Protenix v2 today | with a drifting head |
|---|---|---|
| network evaluations per sample | $N_{\text{step}}=200$ (config default) | **1** |
| samples per prediction | $N_{\text{sample}}=5$ (config default) | 5, at 1/200 the cost each |
| sampler latency | dominates inference for large complexes | trunk-dominated |
| extra models needed | — | — (no teacher, no distillation) |
| conditioning path | unchanged | unchanged |

The prize is a ~$200\times$ reduction in the sampler's share of inference cost, with no teacher model and no
adversarial training. For screening workloads — thousands of binder candidates against one target — that changes
what is affordable, and it is the strongest argument for doing this at all.

**The central risk, stated plainly.** Structure prediction is the **single-positive regime**: one input, one
reference structure in the PDB. Notebook 8 §4 measures what drifting's field does there — the fixed point is the
reference, and a 32-member ensemble collapses from spread 0.86 Å to 0.004 Å. Diffusion has the same nominal problem
but escapes it because its training objective is denoising at *many noise levels*, which forces the model to
represent a whole family of perturbed structures; drifting has no such mechanism. **A naive port will produce five
identical samples.** Everything in §8 exists to address this, and if §8 fails then Design A is a fast deterministic
predictor and nothing more — which may still be useful, but it is not what the ranking heads expect as input.

**Recommendation.**

1. **Build Design A** (drifting head on a frozen trunk, diffusion module untouched). It is additive, reversible, and
   cheap to falsify. Ship it only if it clears the §12 bar.
2. **Solve diversity via the prior, not the field** — the entropy-scaled prior of `THEORY.md` §13.2, driven by
   trunk-derived per-token uncertainty, plus a positive *set* rather than a positive (§8.2). Test this on a subset
   with genuinely multiple references (NMR ensembles, multiple crystal forms) before scaling.
3. **Do not replace the diffusion module outright** (Design B) until A has demonstrated ensemble diversity. The
   diffusion module is also what feeds the mini-rollout used for confidence and symmetric permutation; removing it
   touches three subsystems at once.
4. **Design C** — drifting as an *ensemble generator* around a diffusion point estimate, with force guidance — is the
   scientifically most interesting option and the one where drifting has an advantage no distillation method has
   (§9, §10). It is a second project, not a prerequisite.

---

## 2. What Protenix's diffusion module actually does

Reading the current code so that the integration points are concrete.

### 2.1 Inference

`protenix/model/generator.py::sample_diffusion` implements AF3 Algorithm 18: a predictor–corrector loop over a noise
schedule, with these config defaults (`configs/configs_base.py`, `sample_diffusion` block):

```
N_step = 200          N_sample = 5
gamma0 = 0.8          gamma_min = 1.0
noise_scale_lambda = 1.003    step_scale_eta = 1.5
```

Each iteration:

1. `centre_random_augmentation` — re-centre and randomly rotate the current coordinates;
2. add noise to move from $\sigma_{i-1}$ to $\hat t$ (the corrector);
3. call `denoise_net` = `DiffusionModule.f_forward` once;
4. take a scaled Euler step to $\sigma_i$.

So the per-sample cost is 200 evaluations of `DiffusionModule`, which is a
24-block/16-head `DiffusionTransformer` over tokens plus 3-block atom encoder/decoder attention
(`protenix/model/modules/diffusion.py::DiffusionModule.__init__` defaults). The trunk (Pairformer, MSA, templates)
runs **once**; the sampler runs 200 times per sample and there are 5 samples.

**This is the cost being attacked.** For a large complex the trunk is expensive, but $200\times5=1000$ diffusion
evaluations are not free, and they scale with atom count through the atom-level attention.

### 2.2 Training

`sample_diffusion_training` does the standard EDM-style step:

1. `centre_random_augmentation(x_gt, N_sample=diffusion_batch_size)` — **48** independently rotated/translated
   copies of the ground truth (config default `diffusion_batch_size = 48`);
2. sample a noise level $\sigma$ per copy from `TrainingNoiseSampler`;
3. one `denoise_net` call on $x_{gt}+\sigma\epsilon$;
4. loss against $x_{gt}$.

The loss (`protenix/model/loss.py`) is a weighted sum with `alpha_diffusion = 4.0`:

* `MSELoss` — aligned coordinate MSE with the EDM $\sigma$-dependent weighting;
* `BondLoss` — bonded-distance error (weight `alpha_bond`, 0 by default, 1 in fine-tuning);
* `smooth_lddt_loss` — a differentiable lDDT surrogate on interatomic distances.

Two structural facts matter for us:

* **The ground truth is randomly re-posed 48 times per step.** The model is *taught* pose invariance by
  augmentation rather than given it architecturally. Any drifting field we add must be equivariant, or it will fight
  this augmentation (Notebook 8 §1).
* **Symmetric permutation is resolved before the loss**, using a 20-step **mini-rollout**
  (`N_step_mini_rollout = 20`, `N_sample_mini_rollout = 1`) whose output is used both to permute the label
  (`permute_label_to_match_mini_rollout`) and to feed the confidence head. The mini-rollout is a *dependency* of two
  other subsystems, not just a sampler.

### 2.3 The integration surface

Everything the diffusion module needs is already computed and passed as arguments:

```
s_inputs [.., N_token, c_s_inputs]   s_trunk [.., N_token, c_s]
z_trunk  [.., N_token, N_token, c_z] pair_z, p_lm, c_l
input_feature_dict (atom_to_token_idx, ...)
```

A drifting head can consume exactly the same conditioning bundle. **That is the single most encouraging engineering
fact in this document**: the change is confined to (a) a new module alongside `DiffusionModule`, (b) a new loss term,
(c) a new sampling entry point alongside `sample_diffusion`. Nothing in the trunk, the featuriser, or the data
pipeline needs to move.

---

## 3. Why drifting is a candidate at all

Against the alternatives for one-step sampling:

| approach | needs a teacher | adversarial | time variable | notes |
|---|---|---|---|---|
| consistency distillation | **yes** | no | yes | inherits the teacher's ceiling; two-stage |
| progressive distillation | **yes** | no | yes | same |
| MeanFlow / shortcut models | no | no | **yes** | still a velocity field over $t$ |
| GAN head | no | **yes** | no | tuning risk on a 3-D constrained output; see Notebook 6 §4 |
| **drifting** | no | no | **no** | needs a metric space; needs diversity solved |

Drifting's specific attractions here:

1. **No teacher.** Distillation caps you at the diffusion module's quality and requires keeping it around. Drifting
   trains from scratch against data.
2. **No time variable.** The entire predictor–corrector schedule, $\gamma_0$, $\lambda$, $\eta_{\text{step}}$
   apparatus disappears. Fewer inference hyperparameters is a real operational win.
3. **Explicit diversity pressure.** The repulsion term is a *built-in* anti-collapse force. In the
   *conditional* setting this does not by itself give aleatoric diversity (§8), but across a batch of different
   inputs it does prevent the degenerate "same fold for every sequence" failure.
4. **Guidance is a training-time property.** `THEORY.md` §10: CFG-like extrapolation is achieved by changing the
   negative pool, at zero inference cost. For structures the analogue would be "extrapolate away from the
   unconditional structure distribution", i.e. sharpen toward the conditioning — with no second forward pass.
5. **Physics enters through the kernel.** `THEORY.md` §12: if you have an energy or forces, you can tilt the field
   toward Boltzmann. No other one-step method has a principled place to put a force field. This is Design C and it
   is the part that is scientifically novel rather than merely faster.

---

## 4. The five risks, stated before the design

Stating these first, because a design that does not address them is not worth reading.

### R1 — The single-positive fixed point (severity: **critical**)

The dataset is (input, one reference structure) pairs. Notebook 8 §4 establishes two things about that regime:

* the naive mean-shift attraction with one positive is $\mathbf{V}^{+} = \mathbf{y}^{\star}-\mathbf{x}$ **exactly**,
  independent of $\tau$ and of the kernel — drifting silently degenerates into MSE regression;
* Drift-React's augmented target set restores $\tau$-dependence, but its fixed point is *still* the reference: a
  32-member swarm collapsed from spread 0.86 to 0.004, and a trained pathway generator learned to ignore its noise
  input entirely.

So: **the equilibrium of a conditional drifting model trained on single references is a deterministic predictor.**
Not a bug to be tuned away — the correct solution to the problem as posed. §8 is entirely about posing a different
problem.

### R2 — Equivariance and permutation (severity: high, but solved)

The field must satisfy $\mathbf{V}(R\mathbf{X}+t)=R\,\mathbf{V}(\mathbf{X})$, or it fights the
`centre_random_augmentation` that already exists. Notebook 8 §3 gives the construction and shows it is exact to
floating point. Additionally, Kabsch needs a consistent atom correspondence, so **symmetric permutation must be
resolved before the field is computed** — the same requirement `SymmetricPermutation` already serves for the
diffusion loss. Solvable, but it means the drifting loss inherits a dependency on the permutation machinery, and
therefore on something like the mini-rollout.

### R3 — The metric space (severity: high)

Notebook 4's lesson: drifting in a space where Euclidean distance is not meaningful does not work. For images that
required an SSL encoder. For structures the candidates are:

* **raw coordinates after Kabsch alignment** — meaningful (it is RMSD), but RMSD is a poor similarity measure for
  large complexes, dominated by whichever domain is largest;
* **pairwise distances on a token subset** — invariant, reference-free, sparse analytic Jacobian (Notebook 8 §5), but
  **achiral** and $O(N^2)$;
* **trunk representations** — $s_{\text{trunk}}$ and $z_{\text{trunk}}$ are already a learned representation of
  structure-relevant features, but they are functions of the *input*, not of the *predicted coordinates*, so they
  cannot discriminate between two candidate structures for the same input. Unusable as-is.
* **a structure encoder trained for the purpose** — the honest answer, and an extra model, which is exactly the
  liability `THEORY.md` §15.4 flags.

Recommendation in §7: a **multi-scale sum** of aligned-coordinate and pairwise-distance terms, with a chirality
term, and *no learned encoder in v1*. Add an encoder only if the ablation demands it.

### R4 — Moving prior and moving metric (severity: high if you co-train)

Notebook 7 §5: if the module producing the prior is trained concurrently, the target moves and training degrades in
proportion to $\rho = \|\Delta m_\psi\| / (\eta\|\mathbf{V}\|)$. Fast co-training measurably damaged a toy that
frozen and staged training handled fine.

For Protenix this is not hypothetical: the trunk *is* a prior-producing module, and if the drifting head is trained
jointly with a trainable trunk, the conditioning distribution moves every step. **Freeze the trunk.** If the trunk
must be fine-tuned later, stage it (trunk frozen → drift head trained → optional joint fine-tune at a low learning
rate with $\rho$ logged).

### R5 — Evaluation (severity: medium, but it is how you avoid fooling yourself)

Notebook 9 §6 documents the per-sample validity crisis: aggregate distance statistics moved from 0.013 to 0.035
while 9.1% of samples were geometrically impossible. For structures the analogues are clashes, bond-geometry
violations, and chirality inversions. **Aggregate lDDT will not show them.** §12 specifies what to measure.

---

## 5. Three architectures

### Design A — drifting head alongside the diffusion module

```
trunk (frozen)  ──►  s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l
                          │
                          ├──►  DiffusionModule  ──► sample_diffusion (200 NFE)   [unchanged]
                          │
                          └──►  DriftingModule   ──► x = f_theta(cond, eps)       [new, 1 NFE]
```

* The drifting head reuses the conditioning bundle verbatim.
* Architecture: reuse `DiffusionModule`'s body — atom encoder, `DiffusionTransformer`, atom decoder — with the
  timestep conditioning path removed and the noise injected as an input perturbation instead.
* Training: new loss term, trunk frozen, diffusion module frozen (or trained as now; they do not interact).
* Inference: a flag selects which sampler runs.
* Confidence head and permutation continue to use the diffusion mini-rollout, so nothing else changes.

**Risk: low. Reversible: completely. This is the one to build.**

### Design B — replace the diffusion module

Same head, but `DiffusionModule` is deleted and the mini-rollout is replaced by a single drifting forward pass. Now
the confidence head is trained on drifting samples and `permute_label_to_match_mini_rollout` consumes a drifting
prediction.

Attractive — the mini-rollout's 20 steps also disappear from *training*, which is a real training-time saving — but
it couples three subsystems to an unproven sampler. **Do this only after A has passed §12**, and note that the
confidence head will need retraining because its input distribution changes.

### Design C — drifting for the conformational ensemble

Keep diffusion for the point estimate. Add a drifting model that, conditioned on the trunk *and on the diffusion
sample*, generates an **ensemble** around it, with the field tilted toward Boltzmann using forces from a neural
network potential or a force field (§9).

This is the option where drifting does something no distillation method can, and where the single-positive problem
partly dissolves: the target is not "the PDB structure" but "the Boltzmann ensemble", and the force labels supply
the ensemble information the PDB does not have. It is also the most research-shaped. §10.

---

## 6. Design A in detail: the one-step drifting head

### 6.1 The generator

$$\mathbf{X} \;=\; f_\theta\big(\boldsymbol\epsilon;\ s_{\text{inputs}}, s_{\text{trunk}}, z_{\text{trunk}}, \text{pair}_z, p_{lm}, c_l\big), \qquad \boldsymbol\epsilon\in\mathbb{R}^{N_{\text{atom}}\times3}$$

Concretely, take `DiffusionModule.f_forward` and change three things:

1. **Remove the noise-level conditioning.** `DiffusionConditioning` currently embeds $\sigma$ via a Fourier
   embedding and adds it to the single representation; delete that path (or, for the entropy-scaled prior of §8.1,
   *replace* it with a per-token prior-width embedding — the plumbing is identical and this is a strong reason to
   keep the conditioning slot rather than remove it).
2. **Remove the EDM preconditioning.** The $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}$ scalings exist to
   condition a denoiser across noise levels. With no noise level they collapse to constants; keep only the
   $\sigma_{\text{data}}=16.0$ output scale so coordinates come out in Ångström-like units.
3. **Feed the prior as the input coordinates.** Where the diffusion module receives $x_{\text{noisy}}$, the drifting
   head receives $\boldsymbol\epsilon$ (or, better, prior mean + noise, §8.1).

Everything else — atom attention encoder, `DiffusionTransformer`, atom attention decoder, the conditioning
projections — is reused unchanged. **The parameter count and per-call FLOPs are essentially identical to one
diffusion step**, which is what makes the 1-vs-200 comparison honest.

### 6.2 The loss

Per training example, with $K$ generated samples and one reference $\mathbf{Y}^{\star}$:

1. draw $K$ priors, one forward pass, giving $\{\mathbf{X}_i\}_{i=1}^{K}$;
2. resolve symmetric permutation of $\mathbf{Y}^{\star}$ against the batch (see §6.4);
3. compute the equivariant field $\mathcal{V}(\mathbf{X}_i)$ (§7);
4. normalise per feature-set and sum over $\tau$;
5. regress: $\mathcal{L}_{\text{drift}} = \frac{1}{K}\sum_i \big\|\mathbf{X}_i - \mathtt{sg}[\mathbf{X}_i + \eta\,\mathcal{V}(\mathbf{X}_i)]\big\|^2$.

Then add the *existing* structural regularisers, which are cheap, well-tested, and directly target R5:

$$\mathcal{L} = \alpha_{\text{drift}}\mathcal{L}_{\text{drift}} + \alpha_{\text{bond}}\mathcal{L}_{\text{bond}} + \alpha_{\text{lddt}}\mathcal{L}_{\text{smooth-lddt}}$$

`BondLoss` and `smooth_lddt_loss` take predicted coordinates and ground truth and are agnostic to how the
coordinates were produced — they should work unmodified. **Keep them.** Notebook 9 §6's lesson is that geometric
validity needs explicit attention; here you already have the terms, and they cost nothing.

A note on why this is not "just MSE with extra steps". If §8 succeeds and the positive is a *set* rather than a
single structure, the kernel selects which reference each sample is pulled toward, and the repulsion keeps samples
from all choosing the same one. If §8 fails, the honest description of $\mathcal{L}_{\text{drift}}$ *is* aligned MSE
plus a small swarm-coupling term, and you should say so rather than dress it up.

### 6.3 The prior

Start with $\boldsymbol\epsilon \sim \sigma_{\text{prior}}\mathcal{N}(0,I)$, centred, with
$\sigma_{\text{prior}}\approx\sigma_{\text{data}}=16$ Å so the input is on the same scale the atom encoder already
expects. Then move to the entropy-scaled prior of §8.1, which is where the diversity has to come from.

### 6.4 Permutation and alignment

Order of operations per step, and it matters:

1. **Permute** symmetry-related chains/atoms of $\mathbf{Y}^{\star}$ to match the prediction. The existing machinery
   uses a mini-rollout; with the diffusion module retained (Design A) you can keep using it, or — cheaper and
   self-consistent — permute against the drifting head's *own* first sample. The latter is a stop-gradient
   assignment, so it adds no gradient path.
2. **Centre** all structures.
3. **Kabsch-align** each $\mathbf{X}_i$ onto $\mathbf{Y}^{\star}$, with the $\det=+1$ guard (Notebook 8 §2 — without
   it you will align onto mirror images and never see it).
4. Compute the field in the aligned frame; rotate back by $R_i^{\top}$.

A subtlety worth flagging: for multi-chain complexes a **global** Kabsch alignment is dominated by the largest chain,
so the field will be near-blind to the pose of a small ligand or peptide. Two mitigations: compute the field
per-chain in per-chain aligned frames and sum, or use the pairwise-distance feature set (§7.2), which needs no
alignment at all. Both, ideally — they are different feature sets in the multi-scale sum, which is exactly what the
normalisation machinery of `THEORY.md` §9.2 is designed to combine.

### 6.5 Batch structure

`THEORY.md` §5.4: the batch *is* the ensemble. Reuse the existing `diffusion_batch_size = 48` slot as $K$, the
number of generated samples per input per step — the memory profile is then identical to today's training step,
which is a large practical convenience.

Also: **repel across inputs, not only within them.** Samples for input $a$ and input $b$ should repel each other in
the shared feature space. This does not pin the conditional spread (nothing does — R1) but it prevents the model
from mapping different inputs to the same structure, which is the cross-conditional collapse that would otherwise be
unpenalised. Cost: the affinity matrix becomes $(B\cdot K)^2$ instead of $B\cdot K^2$; at $B=1$ per device this is
free.

### 6.6 Diagnostics — non-negotiable

Notebook 3 §7's four numbers, per feature set and per $\tau$:

1. **raw $\|\mathcal{V}_\tau\|^2$** — the residual force; must fall. Log the *un-normalised* value; the training
   loss is pinned at $\approx\eta^2$ and tells you nothing (§6.2 of `PHYSICS.md`).
2. **kernel ESS** $=1/\sum_j\bar k_j^2$ — the number of structures actually contributing. ESS $\to1$ means the field
   is copying the nearest reference (memorisation); ESS $\to$ all means it is matching moments only.
3. **$\|\mathcal{V}^{+}\|/\|\mathcal{V}^{-}\|$** — should approach 1 at equilibrium. Deviation means attraction and
   repulsion are unbalanced and the anti-symmetry is effectively broken.
4. **ensemble spread** — mean pairwise RMSD among the $K$ samples for one input. **This is the R1 detector.** If it
   falls monotonically to zero, §8 has failed and you should know within a few thousand steps rather than at the end
   of a training run.

Plus two structure-specific ones:

5. **equivariance residual**, computed once at startup and after any change to the field code:
   $\|\mathcal{V}(R\mathbf{X}+t) - R\mathcal{V}(\mathbf{X})\|/\|\mathcal{V}\|$ should be $\sim10^{-6}$ in float32
   (Notebook 8 §3). Anything larger is a dropped frame.
6. **$\rho$ = (conditioning motion)/(drift motion)** if anything upstream is unfrozen (Notebook 7 §5).

---

## 7. The field, concretely, for structures

### 7.1 The construction

Per input, with $K$ generated structures and a positive set $\mathcal{P}$ (§8.2; $|\mathcal{P}|=1$ in the naive
case), following Drift-React (`THEORY.md` §11) generalised to $|\mathcal{P}|\ge1$:

1. Kabsch-align each $\mathbf{X}_i$ onto a canonical reference, giving $R_i$, $\tilde{\mathbf{X}}_i$; align the
   members of $\mathcal{P}$ to the same canonical frame.
2. Build the augmented target set $\mathcal{T} = [\,\mathcal{P},\ \tilde{\mathbf{X}}_1,\dots,\tilde{\mathbf{X}}_K\,]$.
3. Logits $L_{ij} = -d(\tilde{\mathbf{X}}_i, \mathcal{T}_j)/\tau$ where $d$ is the feature-space distance (§7.2).
4. Joint normalisation $A_{ij} = \sqrt{\sigma_{\text{row}}(L)_{ij}\,\sigma_{\text{col}}(L)_{ij}}$ — this is what
   makes $\tau$ meaningful when positives are few (Notebook 8 §4).
5. Split into $A^{+}$ (positive columns) and $A^{-}$ (sample columns), mask the diagonal of $A^{-}$
   (self-exclusion — at $K=48$ this is a 2% bias, at $K=8$ it is 12%).
6. Form the field, rotate back by $R_i^{\top}$.

### 7.2 Feature sets

Do not pick one. Use the multi-scale machinery (`THEORY.md` §9), which normalises each feature set to a common scale
and sums, so that no set dominates by having larger units:

| set | what it measures | invariance | cost | notes |
|---|---|---|---|---|
| **aligned coordinates** | RMSD after superposition | equivariant via Kabsch | $O(K^2N)$ | dominated by the largest chain |
| **per-chain aligned coordinates** | per-chain RMSD | equivariant per chain | $O(K^2N)$ | fixes the domination problem |
| **pairwise distances (subset)** | contact geometry | invariant, no alignment | $O(K^2M)$ for $M$ pairs | achiral; subsample pairs |
| **interface distances only** | binding-site geometry | invariant | small | the one that matters for binders |
| **backbone dihedrals** | local conformation | invariant, **chiral** | $O(K^2N)$ | restores what distances lose |

The last row is the answer to the chirality problem of Notebook 8 §5: pairwise distances cannot see mirror images,
so a distance-only field gives zero signal on chirality. Dihedrals are cheap, invariant, and signed.

For the pairwise-distance sets, subsample: at $N=2000$ atoms the full set is $2\times10^6$ pairs. Sample a fixed
random subset of pairs per step (re-drawn each step, so the expectation covers all of them), or restrict to pairs
within a distance cutoff in the *reference*, which concentrates the signal on contacts.

### 7.3 Choosing $\tau$

After feature normalisation, $\tau$ is dimensionless (`THEORY.md` §9.2): "close" means a fraction of the typical
inter-structure distance. Use the paper's approach — a fixed small set spanning scales, e.g.
$\tau\in\{0.05, 0.2, 0.8\}$ — and **do not tune it per dataset**. Tune it once, by looking at ESS: you want the
smallest $\tau$ to give ESS comfortably above 1 and the largest to give ESS well below $|\mathcal{T}|$.

For structures there is a natural physical reading worth using as a sanity check: a $\tau$ corresponding to $\sim1$ Å
RMSD is "the same structure, different refinement"; $\sim3$ Å is "the same fold"; $\sim8$ Å is "the same rough
topology". You want at least one $\tau$ in each regime.

### 7.4 Equivariance is not enough

Notebook 8 §6 measures this: the *field* being equivariant does not make the *model* equivariant. The trained
generator's equivariance residual there was 0.78, not $10^{-6}$. Protenix already handles this with
`centre_random_augmentation` during training, so the drifting head should keep the same augmentation. The equivariant
field ensures the *target* is well-defined; the augmentation teaches the network.

---

## 8. Where diversity comes from

**This is the section that decides whether the project works.** R1 says the field's fixed point under a single
positive is the reference. So diversity must come from somewhere the field is not.

Four mechanisms, in decreasing order of my confidence in them:

### 8.1 Entropy-scaled priors (recommended, highest confidence)

From Drift-AR (`THEORY.md` §13.2, Notebook 7 §6): tie the per-token prior width to the model's own uncertainty,

$$\boldsymbol\epsilon^{(r)} \sim \mathcal{N}\big(\mu^{(r)},\ \sigma^2(\mathcal{E}^{(r)})I\big),
\qquad \sigma(\mathcal{E}) = \sigma_{\max}\frac{e^{\mathcal{E}/\tau_\sigma}-1}{e^{\mathcal{E}_{\max}/\tau_\sigma}-1}.$$

Where confident, $\sigma\to0$, the prior is a point mass, and the drift there is essentially zero *by construction* —
Notebook 7 §6 measured a $107\times$ ratio in field magnitude between confident and uncertain positions with no
per-position weighting in the loss at all. Where uncertain, the prior is broad and the generator has room to produce
genuinely different structures.

**Protenix already computes exactly the uncertainty signal this needs.** The confidence head produces per-token pLDDT
and PAE. Those are calibrated (that is their entire purpose), they are available before sampling, and they mean
precisely "how much do I know about where this token goes". Use predicted pLDDT (or a monotone function of it) as
$\mathcal{E}^{(r)}$: high-confidence tokens get a tight prior around the trunk's implied position; flexible loops and
disordered termini get a broad one.

This is the single best idea in this document, because it does three things at once:

* it supplies conditional diversity **exactly where the data says diversity exists**, rather than uniformly;
* it makes the drifting head's job easier — most tokens are nearly determined, so the field only has to work on the
  uncertain minority;
* it is *interpretable*: the ensemble spread is, by construction, a function of predicted confidence, which is a
  property a structure-prediction system should have anyway.

The prerequisite is calibration. An overconfident confidence head zeroes the prior at positions that genuinely needed
sampling, and drifting cannot recover what the prior never proposed. Measure calibration on a held-out set *before*
relying on this.

A practical note on plumbing: §6.1 suggested keeping `DiffusionConditioning`'s embedding slot and feeding the
per-token prior width into it instead of $\sigma$. That is not an accident — the shapes match, and the network then
*knows* which tokens have wide priors, which is exactly what it needs to know.

### 8.2 A positive *set* instead of a positive (recommended, medium confidence)

Drifting's mechanism needs several positives to be a kernel method at all (Notebook 8 §4). The PDB does contain
multiple structures for many entities:

* **NMR ensembles** — an explicit conformational ensemble, already deposited as multiple models;
* **multiple crystal forms / multiple entries** for the same sequence, and multiple copies in one asymmetric unit;
* **apo/holo pairs** and different ligand-bound states;
* **near-identical sequences** (>95% identity) as approximate positives;
* **MD snapshots**, where available, for entities that have been simulated.

Build a positive set $\mathcal{P}$ per input by clustering these. Then the field is a *real* kernel method: sample
$i$ is pulled toward whichever reference it is closest to, repulsion keeps samples from all choosing the same one,
and $\tau$ decides how sharply. This is the regime the whole method was designed for.

Coverage is the issue — a minority of entities will have $|\mathcal{P}|>1$. Two responses: (a) it is still a strictly
better training signal for that minority, and they are disproportionately the flexible entities where diversity
matters; (b) train with a mixed batch and accept that the single-positive examples contribute a regression signal
while the multi-positive ones contribute a distributional signal. That is not elegant but it is honest, and the
diagnostic in §6.6(4) will tell you whether the multi-positive examples are enough to keep the model from collapsing
globally.

### 8.3 Stochastic conditioning (medium confidence, cheap)

Protenix's trunk already has a stochastic input: **MSA subsampling**. Different MSA samples produce different trunk
representations and, in AlphaFold-family models, are a known lever for generating alternative conformations. If the
conditioning bundle differs between samples, the drifting head's outputs differ even if the head is deterministic
given its conditioning.

This is nearly free — it is a data-pipeline change, not a modelling one — and it composes with §8.1. The caveat is
that it moves the diversity question upstream: the ensemble you get reflects MSA-subsampling variance, which is not
the same as the Boltzmann ensemble and should not be described as such.

### 8.4 An explicit diversity term (low confidence, use as a diagnostic)

Add a term rewarding spread among the $K$ samples for one input — effectively, an unmatched repulsion. This will
certainly produce spread. It will just as certainly produce spread *in the wrong directions*, because nothing tells
it which degrees of freedom are the flexible ones; you will get inflated RMSD from rigid regions wobbling.

Use it as a diagnostic (does the model *can* produce diverse outputs, or has it lost the capacity?), not as a
solution.

### 8.5 What to conclude if all of this fails

Then Design A yields a **fast deterministic structure predictor**: one forward pass, no sampler, a single output per
input. That is genuinely useful for screening, and it should be reported as what it is rather than as a generative
model with five identical samples. It would also mean the confidence head — which is trained on a *distribution* of
rollouts — needs rethinking, since its input distribution would have collapsed.

---

## 9. Physical guidance from energies and forces

This is where drifting can do something the alternatives cannot, and Notebook 9 establishes both the mechanism and
its hazard.

### 9.1 Which mechanism: FK, not FI

`THEORY.md` §12.6 and Notebook 9 §6, measured:

* **FI (force interpolation)** changes the drift *direction* by blending in $\tau^2\mathbf{F}/k_BT$. In a constrained
  output space this leaves the data manifold: in distance space the rank-3 embedding residual tripled and bond
  lengths were perturbed by 0.27 Å at $\omega=1$.
* **FK (force-aligned kernel)** changes only the *weights*, $\ell_j \mathrel{-}= \gamma\mathcal{E}(\mathbf{y}_j)/k_BT$.
  Displacements remain differences of real structures, so validity is preserved by construction: embedding residual
  and bond error both essentially unchanged.

> **For structure prediction, use FK.** Valid protein structures are a thin manifold in $\mathbb{R}^{3N}$;
> direction-space guidance will leave it, and the failure is per-sample and invisible in aggregate metrics.

### 9.2 What to tilt with

$\ell_j \mathrel{-}= \gamma\,\mathcal{E}(\mathcal{T}_j)/k_BT$ over the augmented target set, where $\mathcal{E}$ can be:

* a **neural network potential** evaluated on the candidate structures — the natural choice, differentiable, but
  a per-step cost on $K$ structures;
* a **classical force field** energy (or just its clash/steric terms) — cheap, and clashes are the failure mode most
  worth targeting;
* a **learned scoring function** — anything monotone in "how physical is this" works, since only differences within
  the softmax matter and $\log Z$ cancels.

The tilt is applied to the *repulsion* side as well as the attraction if you want the model's own high-energy samples
downweighted — but note that doing so breaks anti-symmetry in a second place, and §12.5 of `THEORY.md` says the
equilibrium then moves in a way that is harder to predict. Start with attraction only.

### 9.3 What to expect

* **ESS falls as $\gamma$ rises** (Notebook 9 §4: 60000 → 30000 for a modest tilt on a 2-D problem). With $K=48$
  structures, even a moderate tilt can leave an effective handful. Log ESS *with the tilt applied*; if it approaches
  1, the tilt has become an argmax over the batch.
* **The equilibrium moves but does not arrive.** Force guidance shifts the ensemble between $p_{\text{data}}$ and
  $p_{\text{Boltz}}$ and reaches neither (Notebook 9 §5). Report distance to both. "Physically better than the PDB
  ensemble" and "Boltzmann-correct" are different claims and only the first is available.
* **A units trap.** Notebook 9 §3 documents the specific bug: normalising coordinates before evaluating
  $\mathbf{F}$ or $\mathcal{E}$ evaluates them at the wrong physical point, and the symptom is that the guidance
  moves the ensemble the *wrong way*. Energies must be evaluated in Ångströms on unnormalised coordinates.

---

## 10. Design C: drifting for the conformational ensemble

The most interesting version, sketched because it changes what the system is for.

**Setup.** Condition on the trunk *and* on a diffusion-generated point estimate $\mathbf{X}_0$. The drifting head
generates $K$ structures around $\mathbf{X}_0$. The positive set is whatever ensemble information exists (§8.2), and
the field is FK-tilted by a potential (§9). The prior is entropy-scaled (§8.1), so spread is concentrated where the
model is uncertain.

**Why the single-positive problem is weaker here.** The target is no longer "the deposited structure" but "the
Boltzmann ensemble at 300 K", and the force labels carry ensemble information that no PDB entry contains. §12.2 of
`THEORY.md`: the Boltzmann-weighted attraction is computable *entirely from forces*. You are no longer trying to
extract a distribution from one sample; you are using the energy function to define the distribution and using the
sample to anchor it.

**Why this is the scientifically distinctive option.** Every other one-step method distils a sampler that was itself
trained only on deposited structures, so its ensemble can only ever reflect PDB statistics. Drifting with force
guidance can, in principle, produce an ensemble the training data never contained. That is a different claim from
"faster", and a much stronger one — with a correspondingly higher burden of proof (§12.4).

**Cost.** A potential evaluation on $K$ structures per step. With a small NNP this is comparable to one diffusion
step; with a large one it dominates. Precompute forces on the training structures where possible.

---

## 11. Training plan, staging, and compute

Notebook 7 §5 argues for staging; here is the ladder. Each rung has a decision point.

**Stage 0 — plumbing, no learning.** Implement the field; verify equivariance to $10^{-6}$ in float32; verify the
$\det=+1$ guard; verify $\tau$-dependence is non-trivial (Notebook 8 §4 — if the field is $\tau$-inert you have the
single-positive degeneracy and the augmented target set is not wired up). *Decision: does the field behave?*

**Stage 1 — one protein, overfit.** One input, $K$ samples, frozen trunk. The model should reproduce the reference.
Watch the ensemble-spread diagnostic collapse — **you want to see this**, because it confirms R1 is real in your
setup and gives you the baseline that §8 must beat. *Decision: does the loss drive RMSD down at all?*

**Stage 2 — small set, single positives.** A few hundred monomers. Now measure the ensemble-spread trajectory
properly, and the four diagnostics. Expect a deterministic predictor. *Decision: what accuracy does a 1-NFE
deterministic drifting head reach, relative to the diffusion module?* This number alone justifies or kills Design A
as a screening tool, independent of diversity.

**Stage 3 — diversity mechanisms.** Add §8.1 (entropy prior from pLDDT) and §8.2 (positive sets) on a subset chosen
to *have* multiple references — NMR ensembles first, since the ground-truth diversity is explicit. *Decision: does
the ensemble spread track the reference ensemble's spread, per residue?* This is the make-or-break measurement of the
whole project.

**Stage 4 — scale.** Full dataset, batch structure as §6.5, EMA (Notebook 3 §6 — the single largest measured
improvement in the toy setting; match the horizon to the run length, do not copy a decay constant from a paper),
warmup, gradient clipping at 2.0.

**Stage 5 — optional joint fine-tune.** Unfreeze the trunk at a low learning rate, logging $\rho$ (§6.6). If $\rho$
exceeds $\sim0.1$, slow down or stop. Expect small gains; this is the highest-risk, lowest-reward rung.

**Compute.** Per training step the drifting head costs $K$ forward passes plus one backward, versus the diffusion
module's $\text{diffusion\_batch\_size}=48$ forward passes plus one backward — i.e. **comparable at $K=48$**, plus
the field computation, which is $O(K^2)$ distance work and negligible against transformer FLOPs. Training cost is
therefore in the same ballpark as the existing diffusion training, and Design B would actually be *cheaper* because
it removes the 20-step mini-rollout from training.

---

## 12. Evaluation protocol

### 12.1 Accuracy (versus the diffusion module, same trunk, same test set)

* lDDT, per-complex and per-interface; DockQ for interfaces; TM-score for folds.
* Report as a **paired** comparison on identical inputs. The trunk is frozen and shared, so the only variable is the
  sampler.

### 12.2 Per-sample validity — R5, and non-negotiable

Reported as **fraction of samples passing**, never as a mean (Notebook 9 §6: a mean over a bimodal validity
distribution describes nothing):

* steric clashes per structure, and fraction of structures with zero clashes;
* bond lengths and angles within tolerance;
* **chirality**: fraction of C$\alpha$ centres with correct handedness — this is the one that a distance-based
  feature set is blind to, so it is the one most likely to regress silently;
* peptide-bond geometry, cis/trans;
* ring planarity for aromatics.

Compare against the diffusion module on the same inputs. A 1-NFE sampler that is 0.5 lDDT better and produces 5%
invalid structures is worse, not better.

### 12.3 Ensemble quality

* **spread**: mean pairwise RMSD among the $K$ samples, per input, and its trajectory over training;
* **calibration of spread**: correlation between per-residue ensemble spread and per-residue error against the
  reference. A good ensemble is uncertain where it is wrong. This is the metric that distinguishes real aleatoric
  diversity from noise, and it is the one to lead with;
* on NMR-ensemble targets: does the predicted per-residue spread track the deposited ensemble's?
* **coverage**: for entities with known alternative conformations, does any of the $K$ samples find the alternative?

### 12.4 If force guidance is used (Design C)

* distance to the PDB ensemble **and** distance to the Boltzmann ensemble, both reported (Notebook 9 §5);
* effective sample size of the tilted kernel;
* energy distribution of generated structures versus MD reference, if available.

### 12.5 Cost

Wall-clock and FLOPs for sampler-only and end-to-end, at several complex sizes. The headline claim is a cost claim;
measure it properly, and separate trunk cost from sampler cost so the improvement is not diluted by the part that did
not change.

---

## 13. Risk table and kill criteria

| # | risk | severity | evidence | mitigation | kill criterion |
|---|---|---|---|---|---|
| R1 | single-positive fixed point = the reference; no ensemble diversity | **critical** | Notebook 8 §4; spread 0.86 → 0.004 | §8.1 entropy prior, §8.2 positive sets | after Stage 3, ensemble spread does not correlate with per-residue error, and does not track NMR ensemble spread |
| R2 | equivariance / permutation bugs | high | Notebook 8 §§2–3 | exact construction + startup assertion | equivariance residual $>10^{-4}$ in float32 and not traceable |
| R3 | no adequate metric space for structures | high | Notebook 4 | multi-scale feature sets, §7.2 | ESS stuck at 1 or at $|\mathcal{T}|$ across all $\tau$ after normalisation |
| R4 | moving prior/metric if co-trained | high | Notebook 7 §5 | freeze the trunk; stage; log $\rho$ | $\rho>1$ and TV/lDDT non-monotone |
| R5 | per-sample validity regressions hidden by aggregate metrics | medium | Notebook 9 §6 | keep BondLoss + smooth-lDDT; per-sample reporting | validity fraction below the diffusion module's at equal lDDT |
| R6 | accuracy simply lower than 200-step diffusion | medium | — | more capacity; more $\tau$; feature encoder | Stage 2 lDDT gap $>$ 3 points and not closing with scale |
| R7 | confidence head degrades (Design B) | medium | mini-rollout dependency | keep diffusion mini-rollout (Design A) | — |
| R8 | force guidance leaves the structure manifold | medium | Notebook 9 §6 | **use FK, not FI** | validity regresses with $\gamma$ |

**The one measurement that decides the project** is R1's kill criterion. Everything else is engineering.

---

## 14. Open questions

1. **Does the augmented-target-set field recover aleatoric diversity when positives are genuinely multiple?**
   `THEORY.md` §15.7 lists this as open, and Notebook 8 only tests the single-positive case. The NMR-ensemble subset
   is the cleanest available test and it should be run early — it is a small experiment that determines whether the
   large one is worth doing.

2. **What is the right metric for structures?** RMSD is dominated by size; distances are achiral; lDDT is not a
   metric. Is there a kernel on structures for which the drifting field's equilibrium is the right object? This is
   `THEORY.md` §15.3 specialised to geometry, and it is a genuinely open modelling question rather than an
   engineering one.

3. **Can the confidence head be trained on drifting samples?** It currently consumes a mini-rollout, i.e. a
   distribution over partially-denoised structures. A 1-NFE sampler gives a different — and if R1 bites, degenerate —
   input distribution. This may be the real obstacle to Design B.

4. **Does CFG-as-negative-mixture (`THEORY.md` §10) have a useful structural analogue?** "Extrapolate away from the
   unconditional structure distribution" would sharpen predictions toward the conditioning at zero inference cost.
   What is $p_{\text{data}}(\cdot|\varnothing)$ for structures — a fold-frequency prior? Random structures from the
   PDB? The answer determines whether this is a real lever or a category error.

5. **Is drifting stable at Protenix's scale?** The largest reported drifting model is 464M parameters on ImageNet.
   Protenix-v1 base is 368.5M parameters in total (`configs/configs_model_type.py` table; check the v2 figure), of
   which the diffusion module is a substantial fraction — applied to a far more constrained output space, with batch
   sizes set by memory rather than by field-estimator variance. `THEORY.md`
   §15.5 notes that drifting's scaling behaviour in batch size is largely uncharted, and structure prediction pushes
   on exactly that axis.

---

## 15. Empirical update: what picodrift measured

Sections 1–14 were written before any experiment. `picodrift/` has since tested several of their claims on the
smallest real system that has the same structure as the problem. It replaces picofold's flow matching with drifting
for 16-residue Cα fragments from CATH S40, with one reference structure per sequence, in one harness with matched
compute. The numbers below are from `picodrift/results/`. The method and caveats are in
`picodrift/reports/picodrift_report.md`. The scale is tiny (1M parameters, local structure only), so read these as
directional evidence, not as Protenix measurements.

| claim in this document | what picodrift measured |
|---|---|
| **R1** (§4): single-positive drifting reduces to a deterministic predictor | **Confirmed, in a sharper form.** With a capacity-limited network the failure is *over-dispersion*, not collapse: lDDT 0.449 against flow's 0.621, below a random real fragment (0.522), with ensemble spread 4.5 Å and 72% clashes. The shared network reconciles single positives across sequences it cannot tell apart, because sequence barely predicts local structure here, and settles on an over-dispersed blob. A 2-D toy (notebook 11 §7) shows the regime dependence: one positive per input is nearly harmless when the input determines the output, and costly when it does not. When the network *can* separate every sequence (512 training windows) the field collapses each ensemble onto its reference, exactly as stated here (training lDDT 0.994). |
| §8.5: if diversity fails, Design A is still a fast deterministic predictor | **Not reached.** Naive drifting is worse than plain aligned regression (0.466) at the same cost. |
| §8.2: a positive *set* instead of a positive | **The decisive lever, but only if the set carries information.** Sixteen samples per training sequence from a trained flow model (one offline pass, 630 s here) remove the over-dispersion (spread 2.7 Å). With geometry fixed (next row) the one-step student reaches **lDDT 0.612 ± 0.000 against its 200-step teacher's 0.621 ± 0.001**, about 170× cheaper per structure. Sets of the same size *without* new information do not help: noisy copies of the reference score 0.540, and references of embedding neighbours 0.530. Sequence-similarity neighbours are barely informative here (BLOSUM62 top-5: 0.563). Adding the true reference to the distilled set *hurts* (0.529 vs 0.542). |
| §6.2: keep `BondLoss`; geometry needs explicit attention (R5) | **Stronger than stated: enforce it by parameterisation.** An internal-coordinate output, with bonds fixed at 3.8 Å, plus a clash penalty, was worth as much as the positive set (+0.10 lDDT each, +0.163 together). Bond error fell from 1.10 to 0.023 Å, below the flow teacher's 0.083. |
| §12.2: aggregate lDDT will not show per-sample failures | **Confirmed with a sharper example.** An internal-coordinate regressor scored a respectable lDDT (0.556) while 61% of its samples were *mirror images*. lDDT and dRMSD are reflection-invariant, so measure chirality explicitly. |
| Q4 (§14): CFG as a negative mixture for structures | **A real lever for a broken field, idle for a good one.** Using other sequences' references as negatives gives +0.04 to +0.05 over naive drifting (3 seeds, α ∈ [1.2, 2]), and nothing once distilled positives are used (0.613 vs 0.612). |
| One step versus few steps | **One call is enough.** N drifting stages along the flow path never beat the single jump: 0.611 (N = 1), 0.613 (N = 2), 0.594 (N = 4), 0.579 (N = 8), and 0.610 for N = 4 trained on its own rollouts. Without a teacher, where the noisy intermediates at least supply positive sets, it is the same: 0.556 (N = 1) against 0.530 (N = 4) and 0.499 (N = 8). A few-step drifting head adds cost, not quality. |

**What this changes in the design.**

* **Design A should be trained as a distillation of the diffusion module from the start.** The step is: sample the
  existing module over the training set once, and drift the head toward those samples, *excluding* the PDB
  reference. This promotes §8.2 from "medium confidence" to the default.
* **Build the head's output so it cannot break the chain.** Use frames or internal coordinates where possible, with
  bond and clash terms on top.
* **Add a chirality check to the §12.2 validity suite.**
* **Do not plan a few-step drifting variant.**
* **Open question.** The 1-NFE head's quality is capped by its teacher. Whether the drifting field adds anything over
  plain regression onto teacher samples has not yet been measured; it is the next picodrift control.
