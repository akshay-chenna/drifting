# Learning Drifting Models

A self-contained tutorial on **drifting** — one-step generative modelling by training-time distribution transport —
built from first principles, from the kernel arithmetic up to a concrete engineering proposal for a
structure-prediction system.

Drifting ([Generative Modeling via Drifting](https://lambertae.github.io/projects/drifting/)) generates in **one
network evaluation**. There is no noise schedule, no ODE solver, no time variable, no discriminator, and no teacher
model to distil. Instead, a *drifting field* $\mathbf{V}$ measures the disagreement between the model's samples and
the data, and the model is regressed onto its own samples after they have been nudged along that field. When the two
distributions agree, the field vanishes, the target equals the input, and training has converged.

```python
for step in range(n_steps):
    eps   = randn(N, C)                       # prior noise
    x     = f_theta(eps)                      # generated samples == negatives
    y_pos = sample_data(P)                    # positives

    with no_grad():
        V = sum(normalise(compute_V(x, y_pos, tau=t)) for t in taus) * eta

    loss = mse(x, (x + V).detach())           # regress onto the drifted target
    loss.backward(); opt.step()
```

Sampling is `f_theta(randn(n, C))`. One call. That is the whole method.

This repository builds it up carefully, verifies every claim numerically, and then applies it: the same machinery
appears in *Drift-React* (reaction pathways on SE(3)), *Pepti-drift* (toxicity-repulsive peptide design),
*Drift-AR* (learned priors), and *Drifting to Boltzmann* (force-guided molecular generation). All four are covered.

---

## Documents

| file | what it is |
|---|---|
| **`THEORY.md`** | The mathematics. Anti-symmetry and equilibrium, kernel design, the fixed-point loss and why the stop-gradient is essential, identifiability, gradient-flow theory, the MMD connection, feature-space normalisation, CFG as a negative mixture, SE(3) equivariance, score/force guidance, discrete and latent drifting, a unified comparison table, and the open problems. |
| **`PHYSICS.md`** | The same objects as physics. Drifting as a two-component plasma finding mechanical equilibrium: screening, entropic forces, nonreciprocal interactions, overdamped dynamics and mobility, Henderson's theorem and the identifiability question, force matching and coarse-graining, gauge fixing and equivariance, non-conservative fields, and a renormalisation reading of the multi-$\tau$ sum. Ends with a dictionary. |
| **`PROTENIX_DRIFTING.md`** | A design document: what it would take to replace or augment Protenix v2's diffusion module with a drifting head. Three architectures, five risks stated before the design, the diversity problem and four candidate solutions, an evaluation protocol, and explicit kill criteria. No code. |
| **`REFERENCES.md`** | Source papers. |
| **`picodrift/`** | The method applied, as a production training stack on Modal: picofold's flow matching replaced by drifting, with a head-to-head comparison over four rounds of experiments. A separate git repository with its own `README.md`, `CLAUDE.md` and a paper-style report in `reports/`. |

---

## Notebooks

Each is self-contained and runnable; each verifies its claims numerically rather than asserting them.

### Prerequisites

**`0a. Kernels, Mean Shift, and Vector Fields`** — what a kernel is, kernel density estimation, the mean-shift
vector, attraction and repulsion, flowing points along a field, conservative versus non-conservative fields, and why
softmax normalisation is not optional.

**`0b. Pushforward Maps and Distances Between Distributions`** — the pushforward, measuring the gap between two
distributions from samples alone, the continuity equation and Wasserstein gradient flow, and why batch size matters
so much.

**`0c. Fixed Points, Stop-Gradients, and Self-Distillation`** — fixed-point iteration, why the stop-gradient is
load-bearing rather than an optimisation trick, the value of the loss versus its gradient, and the implicit step
size.

### Core

**`1. Drifting basics`** — the whole method on 2-D data. Anti-symmetry, `compute_V` line by line, the training
objective, one-NFE inference, and reproductions of the paper's anti-symmetry ablation and initialisation-robustness
results.

**`2. Anatomy of the Drifting Field`** — dissecting $\mathbf{V}$: temperature, row versus column normalisation,
positives-versus-negatives allocation, self-exclusion, Laplace versus Gaussian kernels, and the identifiability gap.

**`3. Practical Drifting`** — the recipe that makes it work at scale. Feature and drift normalisation, the
$\eta\times\text{lr}$ valley, the sample queue, EMA, and **the four numbers you should always log**, with a
failure-mode catalogue keyed to them.

**`4. Feature-Space Drifting`** — why Euclidean distance between images is not a similarity, how a self-supervised
encoder fixes it, multi-scale feature sets, and why encoder quality dominates everything else.

**`5. Conditioning and Classifier-Free Guidance`** — conditional drifting, and guidance implemented as a change to
the *negative distribution* at training time — so inference stays at 1 NFE, unlike ordinary CFG's two forward passes.

**`6. Drifting vs Diffusion, Flow, GAN and MMD`** — one target, one architecture, one budget, five methods. Contains
the two structural results: **the gradient of MMD is the un-normalised drifting field** (verified against autograd to
machine precision), and **drifting is a GAN whose discriminator has been replaced by a fixed kernel**.

**`7. Latent Drifting and Learned Priors`** — drifting inside a frozen VAE latent, why equilibrium in latent space is
not equilibrium in data space, $\ell_2$-normalised latents, the **moving prior** problem with a diagnostic ratio that
predicts it, and entropy-scaled priors that make the drift vanish where a predictor is already confident.

**`8. Equivariant Drifting on SE(3)`** — Kabsch alignment with the reflection guard, the equivariant field verified to
floating-point noise, invariant pairwise-distance features with an analytic Jacobian, hard constraints enforced by
output parameterisation, and the uncomfortable result at the centre of scientific applications: **under a single
positive, ensemble diversity is a training budget, not an equilibrium**.

**`9. Score and Force Guidance`** — the Drifting Score Identity (and its failure for the Laplace kernel),
force-interpolated and force-aligned drifting, a quantitative test of the predicted temperature-tilt exponent
$\gamma^\star$, and the representation reversal: **modifications that change direction can leave the data manifold;
modifications that change only weights cannot**.

**`10. Discrete Drifting for Peptides`** — drifting in a learned spherical latent with a non-autoregressive decoder,
objective steering rather than distribution matching, toxicity-repulsive generation, and a measurement of the mode
collapse that follows when repulsion points at a fixed pool instead of at the model's own samples.

### Application

**`11. From Notebooks to picodrift`** — the prerequisites for the application below, on picofold's real data:
protein fragments as data, lDDT and dRMSD and their high floor, why both are exactly blind to mirror images,
virtual torsions and a chirality metric, building chains from internal coordinates (and the sign error that yields
the mirror image at identical lDDT), clash thresholds measured from data, picofold's Euler and Euler–Maruyama
samplers against step count, and a toy of the single-positive conditional regime. That toy shows **one reference
per input is nearly harmless when the input determines the output, costly when it does not, and collapses when the
inputs can be memorised**.

**`picodrift/`** — [picofold](picofold) (a ~200-line flow-matching protein-fragment folder, cloned here as the
baseline) with flow matching replaced by drifting: same data, transformer and evaluation, trained on Modal in one
script with DDP, checkpointing, exact resume and a numerical selftest. It is the first time this tutorial's method
meets real structure data. **Naive drifting fails:** lDDT 0.449 against flow's 0.621, below a random real fragment's
0.522, with over-dispersed rather than collapsed ensembles. The cause is the single-positive regime
(`PROTENIX_DRIFTING.md` R1): with one reference per sequence the attraction is τ-inert while the repulsion stays
local, and with sequences that barely predict local structure the network cannot localise it. A memorisation test
confirms the averaging: on 512 sequences the same field collapses onto the references (train lDDT 0.994). Notebook 11
§7 reproduces all three regimes in a toy. **Fixed, it works:** positive *sets* distilled from the flow model (§8.2 by distillation), plus internal-
coordinate outputs that fix Cα–Cα bonds at 3.8 Å, reach **lDDT 0.612 ± 0.000 in one network evaluation, against
flow's 0.621 at 200**. Two further controls locate the limit. Teacher-free positive sets of the same size (noisy
references, embedding neighbours) do not help, so the repair works through the teacher's information, not the number
of positives. Multi-stage drifting along the flow path does not beat a single jump, so the number of steps is not the
limit either. See `picodrift/README.md` §§4.5–4.7 and 5, and the paper-style write-up in
`picodrift/reports/picodrift_report.md`.

---

## Reading orders

* **I want to understand the method.** `1` → `2` → `3`, then `THEORY.md` §§1–9.
* **I want to know how it differs from what I already know.** `6`, then `THEORY.md` §14.
* **I want the physics.** `PHYSICS.md`, with `0a` and `0c` alongside.
* **I want to apply it to molecules.** `8` → `9`, then `THEORY.md` §§11–12, then `PROTENIX_DRIFTING.md`.
* **I want to apply it to sequences.** `7` → `10`, then `THEORY.md` §13.
* **I want to see it fail on real structures, and why.** `8` §4, then `11`, then `picodrift/README.md` §§4–5, then
  `PROTENIX_DRIFTING.md` §8.
* **I am debugging a run that will not train.** `3` §§7–8, then `PHYSICS.md` §§2.3, 6.2, 11.

---

## What is actually established here

Claims in this repository are measured, not asserted. The load-bearing ones:

| claim | where |
|---|---|
| $\mathcal{L} = \|\mathbf{V}\|^2$ exactly, and the stop-gradient is what makes it so | `0c`, to $10^{-9}$ |
| the Gaussian mean-shift vector is $\tau^2\times$ the score of the KDE | `0a` §6, to $10^{-7}$ |
| breaking anti-symmetry collapses the model | `1` §9 |
| the MMD gradient *is* the un-normalised drifting field | `6` §3, to $10^{-7}$ relative |
| normalisation buys a factor $10^{10}$ in field magnitude 4 units from the data | `6` §3 |
| the Drift-React field is exactly equivariant | `8` §3, $6\times10^{-15}$ (float64), $2\times10^{-6}$ (float32) |
| single-positive attraction is $\tau$-inert — drifting becomes MSE | `8` §4, identical across four decades of $\tau$ |
| the single-positive fixed point is the reference; swarms collapse | `8` §4, spread 0.86 → 0.004 |
| the Drifting Score Identity holds for Gaussian, fails for Laplace | `9` §1, $N^{-1/2}$ vs a 40–60% residual |
| $\gamma^\star = 1/k_BT_{\text{target}} - 1/k_BT_{\text{data}}$ | `9` §4, predicted 0.6, confirmed to $<1\%$ |
| force interpolation degrades geometric validity; kernel reweighting does not | `9` §6, 3$\times$ vs unchanged |
| a moving prior damages training in proportion to $\rho$ | `7` §5 |
| single-positive conditional drifting on real fragments: 200× cheaper, lDDT 0.449 vs flow's 0.621 | `picodrift` §4.1, 3 seeds |
| the single-positive failure is over-dispersion (spread 4.5 Å vs flow's 2.4), not collapse | `picodrift` §§4.1, 5 |
| …and collapse once the network can memorise (512 sequences: spread 6.1 → 0.27 Å, train lDDT 0.994) | `picodrift` §4.5 |
| one positive per input is nearly harmless if the input determines the output (0.164 vs 0.128 from the modes), costly if not (0.327 vs 0.207) | `11` §7 |
| lDDT and dRMSD are exactly blind to mirror images; torsion sines are not | `11` §§2–3 |
| distilled positive sets + internal coordinates: 1-NFE lDDT 0.612 vs flow's 0.621 at 200 NFE | `picodrift` §4.5, 3 seeds |
| more positives without more information do not help: noisy references 0.540, embedding neighbours 0.530 | `picodrift` §4.6 |
| N drifting stages along the flow path never beat one jump: N = 4 independent 0.594 vs N = 1 0.611; rollout 0.610 | `picodrift` §4.7, 3 seeds |

---

## Setup

Python $\ge$ 3.12. Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
jupyter lab
```

Or with pip:

```bash
pip install torch torchvision numpy scipy pandas matplotlib ipywidgets jupyter
```

Notebooks detect CUDA and fall back to CPU. MNIST (notebooks 4 and 7) downloads automatically to `./data`.

The heaviest notebooks are 4 and 7 (a few minutes each on a GPU for the encoder/VAE); everything else runs in
seconds to a couple of minutes. Nothing here needs more than a single consumer GPU.

---

## References

See `REFERENCES.md`. In short:

* **Generative Modeling via Drifting** — the method. <https://lambertae.github.io/projects/drifting/>
* **Drift-React** — SE(3) drifting fields for one-step reaction-pathway generation.
  <https://github.com/schwallergroup/drift-react>
* **Pepti-drift** — toxicity-repulsive drifting for antigen-conditioned discrete peptide generation.
* **Drift-AR** and **Drifting to Boltzmann** — learned priors, and score/force guidance toward the Boltzmann
  distribution.

The companion repository [`learning-diffusion`](../learning-diffusion) covers diffusion models in the same style;
`PHYSICS.md` here is written to be read against its `PHYSICS.md`, since the two methods have genuinely different
physics.
