# The Mathematics of Drifting Models

> Companions: `PHYSICS.md` reads these same objects as an interacting-particle system;
> `PROTENIX_DRIFTING.md` applies them to structure prediction; `README.md` maps the notebooks.

A companion to the [learning-drifting](.) notebooks. This document develops every mathematical idea needed to
understand the code, starting from the definition of a pushforward and building through the drifting field, the
fixed-point objective, identifiability, the gradient-flow picture, feature-space and guided variants, and the
extensions to geometry, discrete data and Boltzmann sampling.

Primary reference: Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He,
*Generative Modeling via Drifting*, [arXiv:2602.04770](https://arxiv.org/abs/2602.04770), 2026.
Equation numbers of the form "(paper Eq. 11)" refer to that paper.

---

## Table of Contents

1. [The problem: matching a pushforward](#1-the-problem-matching-a-pushforward)
2. [Anti-symmetry and equilibrium](#2-anti-symmetry-and-equilibrium)
3. [Designing the drifting field](#3-designing-the-drifting-field)
4. [From a field to a loss: fixed points and the stop-gradient](#4-from-a-field-to-a-loss-fixed-points-and-the-stop-gradient)
5. [Stochastic estimation and Algorithm 2](#5-stochastic-estimation-and-algorithm-2)
6. [Identifiability: does V = 0 imply q = p?](#6-identifiability-does-v--0-imply-q--p)
7. [The gradient-flow picture](#7-the-gradient-flow-picture)
8. [Relation to MMD and moment matching](#8-relation-to-mmd-and-moment-matching)
9. [Feature-space drifting and the normalisation algebra](#9-feature-space-drifting-and-the-normalisation-algebra)
10. [Classifier-free guidance as a negative mixture](#10-classifier-free-guidance-as-a-negative-mixture)
11. [Equivariant drifting on SE(3)](#11-equivariant-drifting-on-se3)
12. [Score and force guidance: drifting toward Boltzmann](#12-score-and-force-guidance-drifting-toward-boltzmann)
13. [Discrete and latent drifting](#13-discrete-and-latent-drifting)
14. [Unified comparison with other paradigms](#14-unified-comparison-with-other-paradigms)
15. [Open problems](#15-open-problems)

---

## 1. The problem: matching a pushforward

We are given samples from an unknown $p_{\text{data}}$ on $\mathbb{R}^{D}$ and want to produce new ones.

Let $f:\mathbb{R}^{C}\to\mathbb{R}^{D}$ be a measurable map and $p_{\boldsymbol{\epsilon}}$ a tractable prior on
$\mathbb{R}^{C}$ (typically $\mathcal{N}(0,I)$, but nothing requires it — the paper also uses discrete "style" indices).
The **pushforward** of $p_{\boldsymbol{\epsilon}}$ under $f$ is the law of $f(\boldsymbol{\epsilon})$:

$$q \;=\; f_{\#}\,p_{\boldsymbol{\epsilon}}, \qquad\text{i.e.}\qquad q(A) = p_{\boldsymbol{\epsilon}}\big(f^{-1}(A)\big) \;\;\text{for measurable } A. \tag{1}$$

Generative modelling is the search for $f$ with $f_{\#}p_{\boldsymbol{\epsilon}}\approx p_{\text{data}}$. Sampling from
$q$ costs exactly one evaluation of $f$. The *density* of $q$ is generally intractable, and — this is the point —
drifting never needs it.

### 1.1 Training-time versus inference-time transport

Deep-learning optimisation is inherently iterative: it produces a sequence of parameter vectors $\{\theta_i\}$, hence a
sequence of maps $\{f_i\}$, hence a sequence of pushforwards

$$q_i = [f_i]_{\#}\,p_{\boldsymbol{\epsilon}}. \tag{2}$$

Fix a noise draw $\boldsymbol{\epsilon}$ and track its image. Between consecutive iterations the sample moves by

$$\Delta\mathbf{x}_i \;:=\; f_{i+1}(\boldsymbol{\epsilon}) - f_i(\boldsymbol{\epsilon}), \tag{3}$$

a displacement caused purely by the parameter update. The paper calls this the **drift**.

This is the conceptual pivot. Diffusion and flow matching also build the transport out of many increments
$\mathbf{x}_{i+1}=\mathbf{x}_i+\Delta\mathbf{x}_i$, but their index $i$ runs over **inference** steps, so every sample
costs $N$ network evaluations forever. Drifting's index runs over **training** steps, so inference costs one
evaluation (1-NFE). Both decompose a hard transport into easy pieces; they differ in *when the bill arrives*.

A corollary worth stating explicitly: a drifting generator has **no time variable**. There is no noise level, no
schedule, no solver, and no self-consistency identity coupling different times. $f_\theta(\boldsymbol{\epsilon})$ is
the sample.

---

## 2. Anti-symmetry and equilibrium

### 2.1 The drifting field

Define a **drifting field** as a map $\mathbf{V}_{p,q}:\mathbb{R}^{d}\to\mathbb{R}^{d}$, depending on two
distributions, that prescribes the training-time displacement:

$$\mathbf{x}_{i+1} \;=\; \mathbf{x}_i + \mathbf{V}_{p,\,q_i}(\mathbf{x}_i), \qquad \mathbf{x}_i = f_i(\boldsymbol{\epsilon}) \sim q_i. \tag{4}$$

We want the field to vanish when the model is correct, so that training has a stationary point at the solution.

### 2.2 Proposition (paper Prop. 3.1)

> If $\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x})$ for all $\mathbf{x}$ (**anti-symmetry**), then
> $q=p \Rightarrow \mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0}$ for all $\mathbf{x}$.

*Proof.* Set $q=p$. Anti-symmetry gives $\mathbf{V}_{p,p} = -\mathbf{V}_{p,p}$, hence $\mathbf{V}_{p,p}=\mathbf{0}$. $\square$

Two remarks.

* **This is the only structural requirement of the framework.** Everything else (mean shift, Laplace kernels,
  softmax normalisation, feature spaces) is a design choice made inside the anti-symmetry constraint.
* **The converse is false in general.** A field may vanish without the distributions matching. §6 treats this.

### 2.3 Anti-symmetry is fragile and load-bearing

The paper's Table 1 breaks anti-symmetry deliberately, scaling the attractive and repulsive halves unequally, on a
B/2 ImageNet model at 100 epochs:

| field | FID |
|---|---|
| $\mathbf{V}^{+}-\mathbf{V}^{-}$ (anti-symmetric) | **8.46** |
| $1.5\mathbf{V}^{+}-\mathbf{V}^{-}$ | 41.05 |
| $\mathbf{V}^{+}-1.5\mathbf{V}^{-}$ | 46.28 |
| $2\mathbf{V}^{+}-\mathbf{V}^{-}$ | 86.16 |
| $\mathbf{V}^{+}-2\mathbf{V}^{-}$ | 112.84 |
| $\mathbf{V}^{+}$ only | 177.14 |

Notebook 1 §9 reproduces the mechanism on a 2-D bimodal target: excess attraction collapses the model to a single
point (sample standard deviation exactly 0), excess repulsion over-disperses it, attraction-only is mode seeking and
collapses. **The repulsion term is what makes drifting a generative model rather than a mode finder.**

---

## 3. Designing the drifting field

### 3.1 The general kernelised form

To be computable from samples, take (paper Eq. 7)

$$\mathbf{V}_{p,q}(\mathbf{x}) \;=\; \mathbb{E}_{\mathbf{y}^{+}\sim p}\,\mathbb{E}_{\mathbf{y}^{-}\sim q}\big[\mathcal{K}(\mathbf{x},\mathbf{y}^{+},\mathbf{y}^{-})\big], \tag{5}$$

with $\mathcal{K}$ a three-point interaction. Any $\mathcal{K}$ giving $\mathbf{V}=\mathbf{0}$ at $p=q$ is admissible.

### 3.2 The mean-shift instantiation

Let $k(\cdot,\cdot)$ be a kernel and define the normalised **mean-shift vectors** (paper Eq. 8–9)

$$\mathbf{V}^{+}_{p}(\mathbf{x}) \;=\; \frac{\mathbb{E}_{p}\big[k(\mathbf{x},\mathbf{y}^{+})(\mathbf{y}^{+}-\mathbf{x})\big]}{Z_p(\mathbf{x})},
\qquad Z_p(\mathbf{x}) = \mathbb{E}_{p}\big[k(\mathbf{x},\mathbf{y}^{+})\big], \tag{6}$$

$$\mathbf{V}^{-}_{q}(\mathbf{x}) \;=\; \frac{\mathbb{E}_{q}\big[k(\mathbf{x},\mathbf{y}^{-})(\mathbf{y}^{-}-\mathbf{x})\big]}{Z_q(\mathbf{x})},
\qquad Z_q(\mathbf{x}) = \mathbb{E}_{q}\big[k(\mathbf{x},\mathbf{y}^{-})\big], \tag{7}$$

and take their difference (paper Eq. 10):

$$\boxed{\;\mathbf{V}_{p,q}(\mathbf{x}) \;:=\; \mathbf{V}^{+}_{p}(\mathbf{x}) - \mathbf{V}^{-}_{q}(\mathbf{x})\;} \tag{8}$$

$\mathbf{V}^{+}_{p}$ is exactly the Fukunaga–Hostetler / Cheng **mean shift** vector: a kernel-weighted average of
displacements toward the data. Alone it is a mode-seeking algorithm. $\mathbf{V}^{-}_{q}$ is the same construction
applied to the model's own samples, and it enters with a minus sign — hence *attraction to $p$, repulsion from $q$*.

### 3.3 The product form and anti-symmetry

The $-\mathbf{x}$ terms cancel when the two expectations are combined, giving (paper Eq. 11)

$$\mathbf{V}_{p,q}(\mathbf{x}) \;=\; \frac{1}{Z_p Z_q}\,\mathbb{E}_{p,q}\Big[k(\mathbf{x},\mathbf{y}^{+})\,k(\mathbf{x},\mathbf{y}^{-})\,(\mathbf{y}^{+}-\mathbf{y}^{-})\Big]
\;=\; \mathbb{E}_{p,q}\Big[\tilde k(\mathbf{x},\mathbf{y}^{+})\,\tilde k(\mathbf{x},\mathbf{y}^{-})\,(\mathbf{y}^{+}-\mathbf{y}^{-})\Big] \tag{9}$$

with $\tilde k = k/Z$ the normalised kernel. Anti-symmetry is now immediate: swapping $p\leftrightarrow q$ exchanges
$\mathbf{y}^{+}\leftrightarrow\mathbf{y}^{-}$, which flips the sign of $(\mathbf{y}^{+}-\mathbf{y}^{-})$ while leaving
the symmetric weight $\tilde k\,\tilde k$ unchanged.

Note the interaction kernel implied by (9),

$$\mathcal{K}(\mathbf{x},\mathbf{y}^{+},\mathbf{y}^{-}) = \tilde k(\mathbf{x},\mathbf{y}^{+})\,\tilde k(\mathbf{x},\mathbf{y}^{-})\,(\mathbf{y}^{+}-\mathbf{y}^{-}), \tag{10}$$

is manifestly anti-symmetric under $\mathbf{y}^{+}\leftrightarrow\mathbf{y}^{-}$. This is the structure §6 exploits.

### 3.4 The kernel

The paper uses a **Laplace** kernel (paper Eq. 12)

$$k(\mathbf{x},\mathbf{y}) = \exp\!\left(-\frac{1}{\tau}\|\mathbf{x}-\mathbf{y}\|\right) \tag{11}$$

with $\|\cdot\|$ the $\ell_2$ distance and $\tau$ a temperature. Because $\tilde k$ is a kernel normalised over the
sample axis, it is literally a softmax over logits $-\|\mathbf{x}-\mathbf{y}_j\|/\tau$ — the same computation as
InfoNCE in contrastive learning.

Why Laplace rather than Gaussian? Two mechanisms:

* **Dynamic range in high dimension.** For roughly isotropic data, $\|\mathbf{x}-\mathbf{y}\|\sim\sqrt{C}$. The
  Gaussian's $\exp(-d^2/2\tau^2)$ therefore collapses to numerical zero far faster with dimension than
  $\exp(-d/\tau)$. A flat kernel is a starved field (§9.3).
* **Discrimination near zero.** The Laplace kernel has a cusp at $d=0$; the Gaussian is flat there. The Laplace
  weights therefore distinguish "on top of a real sample" from "merely near one" much more sharply.

The cost is theoretical: §7 shows the Gaussian field is a gradient field and the Laplace field is not.

---

## 4. From a field to a loss: fixed points and the stop-gradient

### 4.1 The fixed-point relation

At the equilibrium parameters $\hat\theta$ we want $\mathbf{V}_{p,q_{\hat\theta}}\!\left(f_{\hat\theta}(\boldsymbol{\epsilon})\right)=\mathbf{0}$,
which we write as a fixed point (paper Eq. 4):

$$f_{\hat\theta}(\boldsymbol{\epsilon}) \;=\; f_{\hat\theta}(\boldsymbol{\epsilon}) + \mathbf{V}_{p,\,q_{\hat\theta}}\!\big(f_{\hat\theta}(\boldsymbol{\epsilon})\big). \tag{12}$$

The associated iteration (paper Eq. 5) is

$$f_{\theta_{i+1}}(\boldsymbol{\epsilon}) \;\leftarrow\; f_{\theta_i}(\boldsymbol{\epsilon}) + \mathbf{V}_{p,\,q_{\theta_i}}\!\big(f_{\theta_i}(\boldsymbol{\epsilon})\big). \tag{13}$$

### 4.2 The loss (paper Eq. 6)

Realise (13) as a regression with a frozen right-hand side:

$$\boxed{\;\mathcal{L}(\theta) \;=\; \mathbb{E}_{\boldsymbol{\epsilon}}\Big[\big\|\,f_\theta(\boldsymbol{\epsilon}) - \texttt{sg}\big(f_\theta(\boldsymbol{\epsilon}) + \mathbf{V}_{p,\,q_\theta}(f_\theta(\boldsymbol{\epsilon}))\big)\big\|^2\Big]\;} \tag{14}$$

where $\texttt{sg}$ is the stop-gradient. In code:

```python
V      = compute_V(x, y_pos, y_neg)     # no gradient
target = (x + V).detach()
loss   = mse(x, target)
```

### 4.3 Two identities

**Value.** Substituting the target,

$$\mathcal{L}(\theta) = \mathbb{E}_{\boldsymbol{\epsilon}}\big[\|\mathbf{V}(f_\theta(\boldsymbol{\epsilon}))\|^2\big]. \tag{15}$$

The number you print is the squared drift — a genuine measure of distance from equilibrium.

**Gradient.** Writing $\mathbf{x}=f_\theta(\boldsymbol{\epsilon})$ and $\mathcal{L}(\mathbf{x})=\|\mathbf{x}-\texttt{sg}(\mathbf{x}+\mathbf{V})\|^2$,

$$\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\mathbf{x}} = 2\big(\mathbf{x}-\texttt{sg}(\mathbf{x}+\mathbf{V})\big) = -2\,\mathbf{V}(\mathbf{x}), \tag{16}$$

$$\frac{\partial\mathcal{L}}{\partial\theta} = \mathbb{E}\left[\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\mathbf{x}}\,\frac{\partial\mathbf{x}}{\partial\theta}\right] = -2\,\mathbb{E}\left[\mathbf{V}(\mathbf{x})^{\!\top}\frac{\partial\mathbf{x}}{\partial\theta}\right]. \tag{17}$$

**The drift is (minus half) the gradient signal delivered to the output layer.** Equivalently
$\mathbf{V} = -\tfrac12\,\partial\mathcal{L}/\partial\mathbf{x}$, an identity we reuse in §8.

Both identities are verified numerically in Notebook 0c (agreement to $10^{-9}$).

### 4.4 Why the stop-gradient is essential

$\mathbf{V}_{p,q_\theta}$ depends on $\theta$ twice: through its argument, and through the distribution $q_\theta$
supplying the negatives. Differentiating the second dependence is

* **intractable** — in a minibatch the negatives *are* the other samples, so the field couples every sample to every
  other through the softmax normalisation; and
* **the wrong objective** — $\|\mathbf{V}\|^2$ can be driven to zero by moving the *negatives* to meet the query. The
  degenerate optimum is total collapse, where attraction and repulsion are both vacuous.

Notebook 0c demonstrates the second point directly: differentiating the full $\|\mathbf{V}\|^2$ collapses a 2-D model
to a single point (standard deviation exactly $0.000$), while the stop-gradient version recovers $p$.

### 4.5 The implicit step size

A single optimiser step does not reach the frozen target; it closes some fraction $\kappa_i\in(0,1)$ of the gap. The
realised transport per step is therefore $\Delta\mathbf{x}_i \approx \kappa_i\,\eta\,\mathbf{V}(\mathbf{x}_i)$ where
$\eta$ is the explicit drift scale. Transport speed is thus controlled by **two multiplicative knobs**: $\eta$ and the
learning rate. Notebook 3 §4 maps the (wide, diagonal) region in which they cooperate.

---

## 5. Stochastic estimation and Algorithm 2

### 5.1 Minibatch estimator

Draw $N$ noise vectors, form $\mathbf{x}=f_\theta(\boldsymbol{\epsilon})$; these double as the negatives
$\mathbf{y}^{-}$. Draw $N_{\text{pos}}$ real samples $\mathbf{y}^{+}$. Replace the expectations in (9) by empirical
means. The estimator converges at the Monte-Carlo rate $N^{-1/2}$ (verified in Notebook 0b §5), which is why the
paper's Table 2 shows FID improving monotonically in $N_{\text{pos}}$ (20.43 at $N_{\text{pos}}{=}1$ down to 8.46 at 64)
and in $N_{\text{neg}}$ at fixed total batch.

### 5.2 The Algorithm-2 field

The implementation normalises the logit matrix along **both** axes and takes the geometric mean. With
$L\in\mathbb{R}^{N\times(P+M)}$ the concatenated logits $[-d^{+}/\tau \,\|\, -d^{-}/\tau]$:

$$A = \sqrt{\sigma_{\text{row}}(L)\odot\sigma_{\text{col}}(L)}, \tag{18}$$

$$W^{+} = A^{+}\odot\Big(\textstyle\sum_j A^{-}_{\cdot j}\Big),\qquad W^{-} = A^{-}\odot\Big(\textstyle\sum_j A^{+}_{\cdot j}\Big), \tag{19}$$

$$\widehat{\mathbf{V}} = W^{+}Y^{+} - W^{-}Y^{-}. \tag{20}$$

Three observations.

* **This is one Sinkhorn iteration.** Normalising rows and columns pushes $A$ toward doubly stochastic, i.e. toward an
  optimal-transport plan between the query set and the target set. Paper Table 11: no normalisation 10.54 FID,
  row-only 8.92, both 8.46.
* **Anti-symmetry survives.** Exchanging the positive and negative column blocks is a permutation of columns, which
  neither softmax distinguishes; so $W^{+}\leftrightarrow W^{-}$ and $\widehat{\mathbf{V}}\to-\widehat{\mathbf{V}}$.
  Verified exactly (to $10^{-8}$) in Notebook 1 §3.
* **Interpretation as a batch-conditioned field.** The paper writes the double normalisation as a Monte-Carlo estimate
  of (paper Eq. 17)
  $$\mathbf{V}_{p,q}(\mathbf{x}) = \mathbb{E}_{\mathcal{B},p,q}\big[\tilde K_{\mathcal{B}}(\mathbf{x},\mathbf{y}^{+})\,\tilde K_{\mathcal{B}}(\mathbf{x},\mathbf{y}^{-})(\mathbf{y}^{+}-\mathbf{y}^{-})\big],$$
  where $\mathcal{B}$ is the rest of the batch and $\tilde K_{\mathcal{B}}$ normalises using batch statistics. The
  cancellation argument for $p=q$ still goes through term by term.

### 5.3 Self-exclusion

Because $\mathbf{y}^{-}=\mathbf{x}$, the negative distance matrix has zeros on its diagonal — every sample is its own
closest negative with maximal weight. Algorithm 2 masks it (`dist_neg += eye(N)*1e6`). The self term contributes
nothing to the displacement ($\mathbf{x}-\mathbf{x}=\mathbf{0}$) but *steals normalising mass*, shrinking repulsion by
roughly $(1-\tilde k_{\text{self}})$, an $O(1/N)$ bias. It also has a theoretical role: the leave-one-out field is what
removes the reciprocal-KDE self-interaction term from the entropy identity in the finite-particle analysis
([arXiv:2605.22795](https://arxiv.org/abs/2605.22795)).

### 5.4 Batch structure

For class-conditional generation the batch is $N_c$ labels $\times$ $N_{\text{neg}}$ generated samples, with
$N_{\text{pos}}$ positives and $N_{\text{unc}}$ unconditional negatives per label; the field is computed
**independently per label**. Final runs: $N_c{=}128$, $N_{\text{neg}}{=}64$, $N_{\text{pos}}{\in}\{64,128\}$,
$N_{\text{unc}}{=}32$, effective batch $B=8192$. An epoch is $N_{\text{data}}/B$ steps, counted in *generated* samples.

Positives are served from a **MoCo-style sample queue**: a 128-slot FIFO per class plus a 1000-slot global queue,
pushed 64 new samples per step, sampled without replacement.

In a vectorised implementation the per-label independence is obtained by masking: build one $[B, P+B]$ logit matrix
and add $-\infty$ wherever the labels differ. Notebook 5 does this.

---

## 6. Identifiability: does V = 0 imply q = p?

Anti-symmetry gives one direction. The converse is what we actually need, and it is not a theorem — the paper is
explicit that "it remains unclear under what conditions $\mathbf{V}\to\mathbf{0}$ leads to $q\to p$".

### 6.1 The bilinear argument (paper Appendix C.1)

Assume $p$ and $q$ lie in a finite-dimensional model class with linearly independent basis $\{\varphi_i\}_{i=1}^{m}$:

$$p = \sum_{i=1}^m a_i\varphi_i, \qquad q = \sum_{i=1}^m b_i\varphi_i, \qquad \mathbf{a},\mathbf{b}\in\mathbb{R}^m. \tag{21}$$

For a general interaction kernel $K$, $\mathbf{V}_{p,q}(\mathbf{x})=\mathbb{E}_{p,q}[K(\mathbf{x},\mathbf{y}^{+},\mathbf{y}^{-})]$.
Pick $N$ probe locations $\mathcal{X}=\{\mathbf{x}_k\}$ and define the induced interaction vectors

$$\mathbf{U}_{ij}[:,\mathbf{x}] \;=\; \iint K(\mathbf{x},\mathbf{y}^{+},\mathbf{y}^{-})\,\varphi_i(\mathbf{y}^{+})\,\varphi_j(\mathbf{y}^{-})\,d\mathbf{y}^{+}d\mathbf{y}^{-}. \tag{22}$$

Then the field on the probe set is bilinear in the coefficients:

$$\mathbf{V}_{\mathcal{X}} = \sum_{i,j} a_i b_j\,\mathbf{U}_{ij} \;\in\;\mathbb{R}^{d\times N}. \tag{23}$$

Anti-symmetry of $K$ under $\mathbf{y}^{+}\leftrightarrow\mathbf{y}^{-}$ gives $\mathbf{U}_{ij}=-\mathbf{U}_{ji}$ and
hence $\mathbf{U}_{ii}=\mathbf{0}$. Grouping terms over $i<j$,

$$\mathbf{V}_{\mathcal{X}} = \sum_{i<j}\big(a_i b_j - a_j b_i\big)\,\mathbf{U}_{ij}. \tag{24}$$

**Non-degeneracy assumption:** the $\binom{m}{2}$ vectors $\{\mathbf{U}_{ij}\}_{i<j}$ are linearly independent. This
requires diverse probes and $dN \gg m^2$.

Under that assumption, $\mathbf{V}_{\mathcal{X}}=\mathbf{0}$ forces every coefficient to vanish:
$a_ib_j-a_jb_i=0$ for all $i<j$, i.e. $\mathbf{a}\parallel\mathbf{b}$. Since $\int p=\int q=1$ fixes the scale,
$\mathbf{a}=\mathbf{b}$ and $p=q$. Extending to general distributions is a basis-approximation argument:
$\tilde p\approx p$, $\tilde q\approx q$, $\mathbf{V}_{\tilde p,\tilde q}\approx\mathbf{0}$, hence
$\tilde p\approx\tilde q$ and $p\approx q$.

The mean-shift kernel (10) fits this framework with
$K(\mathbf{x},\mathbf{y}^{+},\mathbf{y}^{-}) = k(\mathbf{x},\mathbf{y}^{+})k(\mathbf{x},\mathbf{y}^{-})(\mathbf{y}^{+}-\mathbf{y}^{-})$
provided $Z_p, Z_q$ are finite, so that a vanishing field means a vanishing numerator.

### 6.2 Where identifiability actually fails

The assumption that does the work is the linear independence of the $\mathbf{U}_{ij}$, which is a statement about the
kernel's **resolution**. If the kernel cannot see a difference between two distributions, the field cannot either.

The clean failure mode is $\tau\to\infty$: the field then depends only on the first moment, and *any* $q$ with the
right mean is an equilibrium. Notebook 2 §7 measures this. On a bimodal target with a wrong (unimodal, same-mean) $q$:

| $\tau$ | 0.2 | 0.5 | 1.0 | 3.0 | 10 | 30 |
|---|---|---|---|---|---|---|
| $\|\mathbf{V}\|$ ratio, wrong $q$ vs correct $q$ | 5.4 | 51 | 58 | 11 | 4.5 | **1.3** |

At sensible $\tau$ the field discriminates by 1–2 orders of magnitude; at large $\tau$ it goes blind. Practical
corollaries: always include small $\tau$ in the multi-temperature set, make sure the metric is semantically
meaningful (§9), and monitor the raw $\|\mathbf{V}_\tau\|^2$ *per temperature*, because a run in which only the
large-$\tau$ term is active is matching moments, not distributions.

---

## 7. The gradient-flow picture

### 7.1 The Gaussian score identity

For a Gaussian kernel $K_h(\mathbf{u}) = (2\pi h^2)^{-d/2}\exp(-\|\mathbf{u}\|^2/2h^2)$,

$$\nabla_{\mathbf{x}} K_h(\mathbf{x}-\mathbf{y}) = \frac{\mathbf{y}-\mathbf{x}}{h^2}\,K_h(\mathbf{x}-\mathbf{y}). \tag{25}$$

Let $\rho_{\alpha,h} = K_h * \alpha$ be the kernel density estimate of a measure $\alpha$ and
$s_{\alpha,h}=\nabla\log\rho_{\alpha,h}$ its **smoothed score**. Then

$$s_{\alpha,h}(\mathbf{x}) = \frac{\int \nabla_{\mathbf{x}}K_h(\mathbf{x}-\mathbf{y})\,\alpha(d\mathbf{y})}{\int K_h(\mathbf{x}-\mathbf{y})\,\alpha(d\mathbf{y})}
= \frac{1}{h^2}\,\underbrace{\frac{\int (\mathbf{y}-\mathbf{x})K_h\,\alpha(d\mathbf{y})}{\int K_h\,\alpha(d\mathbf{y})}}_{\text{mean shift } \mathbf{m}_{\alpha,h}(\mathbf{x})}. \tag{26}$$

$$\boxed{\;\mathbf{m}_{\alpha,h}(\mathbf{x}) = h^2\,\nabla\log\rho_{\alpha,h}(\mathbf{x})\;} \tag{27}$$

**The Gaussian mean-shift vector is $h^2$ times the score of the KDE.** Verified to $10^{-7}$ in Notebook 0a §6.
Consequently the Gaussian drifting field is

$$\mathbf{V}^{\text{gauss}}_{p,q}(\mathbf{x}) = h^2\big(s_{p,h}(\mathbf{x}) - s_{q,h}(\mathbf{x})\big) = h^2\,\nabla\log\frac{\rho_{p,h}(\mathbf{x})}{\rho_{q,h}(\mathbf{x})}, \tag{28}$$

a **gradient field** — conservative, curl-free.

### 7.2 Wasserstein-2 gradient flow

Equip $\mathcal{P}_2(\mathbb{R}^d)$ with $W_2$. A curve $\{q_t\}$ with velocity $v_t$ satisfies the continuity equation
$\partial_t q_t + \nabla\!\cdot\!(q_t v_t)=0$. For a functional $\mathcal{F}$, the $W_2$ gradient flow has velocity
$v_t = -\nabla\frac{\delta\mathcal{F}}{\delta q_t}$. For $\mathcal{F}(q)=\mathrm{KL}(q\|p)$ the first variation is
$\log(q/p)+1$, so

$$\boxed{\;v_t(\mathbf{x}) = \nabla\log p(\mathbf{x}) - \nabla\log q_t(\mathbf{x})\;} \tag{29}$$

*Attraction to $p$ minus repulsion from $q$, before any kernel is introduced.* Along this flow
$\mathrm{KL}(q_t\|p)$ decreases monotonically and $q_t\to p$ in $W_2$.

Comparing (28) and (29): with a Gaussian kernel, **drifting is a KDE-approximated Wasserstein-2 gradient flow of
$\mathrm{KL}(q\|p)$**, with time reparameterised by $h^2$ and both scores replaced by kernel-smoothed estimates.
Notebook 0b §4 exhibits the monotone decrease.

This identification is what several follow-ups build on: it is the basis of the *Drifting Field Policy* derivation
([arXiv:2605.07727](https://arxiv.org/abs/2605.07727)), which frames RL fine-tuning of a one-step policy as a reverse-KL
$W_2$ gradient flow toward the soft policy-improvement target $\pi^{+}\propto\pi_{\text{old}}\exp(Q/\alpha)$.

### 7.3 The Laplace field is not conservative

For the Laplace kernel, $\nabla_{\mathbf{y}}k = -\frac{\mathbf{y}-\mathbf{x}}{\tau\|\mathbf{y}-\mathbf{x}\|}k$; the
extra $1/\|\mathbf{y}-\mathbf{x}\|$ breaks (27). The position-dependent normalisation $1/Z(\mathbf{x})$ then destroys
the gradient structure. Numerically (Notebook 0a §6), on the same configuration:

| kernel | mean $|\mathrm{curl}|$ / mean $\|\mathbf{V}\|$ |
|---|---|
| Gaussian | $\approx 10^{-4}$ (finite-difference noise) |
| Laplace | $\approx 1.7\times10^{-2}$ |

So the field the paper actually uses is a genuinely **non-conservative interacting-particle system**. There is no
Lyapunov function handed to you for free, which is why the convergence theory is harder.

### 7.4 Finite-particle convergence

[arXiv:2605.22795](https://arxiv.org/abs/2605.22795) analyses both cases on $\mathbb{R}^d$. Definitions: for particles
$x=(x_1,\dots,x_N)$, $q_x = \frac1N\sum_j K_h(\cdot-x_j)$, $s_x=\nabla\log q_x$, target $\rho=\rho_{\nu,h}$, and

**conservative velocity** $b_x(z) = s_\rho(z)-s_x(z)$, versus
**non-conservative displacement velocity** $u^{\text{disp}}_{\nu,\mu,h}(z) = \mathbf{m}_{\nu,h}(z)-\mathbf{m}_{\mu,h}(z)$
(the paper's field). For Gaussian kernels $b = u^{\text{disp}}/h^2$, so the two coincide up to time rescaling; for
Laplace they differ.

The residual-drift functional is the mean squared speed of the particle cloud,

$$\mathsf{V}_N(x) = \frac1N\sum_{i=1}^N \|b_x(x_i)\|^2, \tag{30}$$

and it controls how much one more drift step would change the generated distribution:

$$W_2\big(\mu^N_x, \widetilde\mu^N_x\big) \le \eta\sqrt{\mathsf{V}_N(x)}, \qquad \widetilde x_i = x_i+\eta\,b_x(x_i). \tag{31}$$

That inequality is the bridge from continuous-time analysis to a **one-step generation guarantee**: a small residual
drift means the one-step sample is already near the fixed point of further refinement.

A joint-entropy identity (extending the finite-particle SVGD analysis) yields, under regularity,
reciprocal-KDE control and quadrature smoothness,

$$\frac1T\int_0^T \mathbb{E}\big[\mathsf{V}_N(X(t))\big]\,dt \;\le\; \frac{H_N(0)}{NT} + \frac{\{-\Delta K(0)\}_{+}\Lambda_T}{N h^{d+2}} + C_T(h)\,h^2. \tag{32}$$

Optimising over $h$ with $T=N$ gives a root residual-velocity rate $N^{-1/(d+4)}$ when the quadrature constant is
$O(1)$, and $N^{-(2-\beta)/(2(d+4-\beta))}$ when it grows like $h^{-\beta}$, $0\le\beta<2$.

For the non-conservative Laplace field, a "sharp companion kernel" $L_h$ with
$\nabla_z L_h(z-y) = (y-z)K_h(z-y)$ gives the decomposition

$$u_x(z) = a_{x,h}(z)\,b^{\#}_x(z) + e_x(z), \tag{33}$$

a positive local scale factor times a sharp-score mismatch, plus a **scale-mismatch residual** $e_x$ that measures the
difference between the data's and the model's local Laplace-weighted radii. The resulting rate carries an
irreducible $\Delta_h^2$ term that vanishes only when those local scales align. **This is the precise price of
non-conservativeness.**

### 7.5 Relation to SVGD

Stein variational gradient descent uses velocity $\mathbb{E}_{q}[k(\mathbf{x},\mathbf{y})\nabla\log p(\mathbf{y}) + \nabla_{\mathbf{y}}k(\mathbf{x},\mathbf{y})]$:
an attraction driven by the *known* target score plus a repulsion from the kernel gradient. Drifting replaces the known
score with kernel-weighted *data samples*, which is what makes it applicable when $p$ is only available empirically.
The entropy-dissipation machinery transfers, which is exactly how (32) is proved.

---

## 8. Relation to MMD and moment matching

### 8.1 Any discrepancy induces a field

By (16), $\mathbf{V} = -\tfrac12\,\partial\mathcal{L}/\partial\mathbf{x}$. So *any* loss $\mathcal{L}$ minimised at
$p=q$ induces a drifting field that vanishes there. The converse fails: not every useful field is the gradient of a
discrepancy (§7.3), and the paper's normalised field is one that is not.

### 8.2 The field behind MMD

With kernel $\xi$,

$$\mathcal{L}_{\text{MMD}^2}(p,q) = \mathbb{E}_{\mathbf{x},\mathbf{x}'\sim q}[\xi(\mathbf{x},\mathbf{x}')] - 2\,\mathbb{E}_{\mathbf{y}\sim p,\mathbf{x}\sim q}[\xi(\mathbf{y},\mathbf{x})] + \text{const}. \tag{34}$$

Differentiating in $\mathbf{x}$ and applying (16),

$$\mathbf{V}_{\text{MMD}}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}^{+}\sim p}\!\left[\frac{\partial \xi(\mathbf{x},\mathbf{y}^{+})}{\partial\mathbf{x}}\right] - \mathbb{E}_{\mathbf{y}^{-}\sim q}\!\left[\frac{\partial \xi(\mathbf{x},\mathbf{y}^{-})}{\partial\mathbf{x}}\right]. \tag{35}$$

For a radial $\xi(R)$, $R=\|\mathbf{x}-\mathbf{y}\|^2$, we get $\partial_{\mathbf{x}}\xi = 2\xi'(R)(\mathbf{x}-\mathbf{y})$ and

$$\mathbf{V}_{\text{MMD}}(\mathbf{x}) = \mathbb{E}_{p}\big[2\xi'(\cdot)(\mathbf{x}-\mathbf{y}^{+})\big] - \mathbb{E}_{q}\big[2\xi'(\cdot)(\mathbf{x}-\mathbf{y}^{-})\big]. \tag{36}$$

Comparing with the mean-shift form (8) identifies the **implied kernel**

$$\tilde k_{\text{MMD}}(\mathbf{x},\mathbf{y}) = -2\,\xi'\!\left(\|\mathbf{x}-\mathbf{y}\|^2\right), \tag{37}$$

which for a Gaussian $\xi$ is $\frac{1}{\sigma^2}\exp(-\|\mathbf{x}-\mathbf{y}\|^2/2\sigma^2)$ — **un-normalised**.

### 8.3 What normalisation buys

The MMD-induced field has no $1/Z(\mathbf{x})$. The consequence is severe far from the data: as
$\|\mathbf{x}\|\to\infty$ every kernel evaluation decays and $\|\mathbf{V}_{\text{MMD}}\|\to0$, so a stranded sample
receives no signal. The normalised field's magnitude instead *grows* with distance, because it is an average of
displacements toward the data. Notebook 0a §7 plots both.

The paper lists four differences and reports being unable to obtain reasonable results with MMD:

1. the $\mathbf{V}$-centric formulation is more general (no discrepancy needs to exist);
2. it supports normalised kernels $\tilde k = k/Z$, which MMD cannot produce;
3. it allows a flexible step size $\mathbf{x}\leftarrow\mathbf{x}+\eta\mathbf{V}$, hence drift normalisation (§9.2);
4. it extends naturally to guidance (§10), which has no MMD analogue.

---

## 9. Feature-space drifting and the normalisation algebra

### 9.1 The formulation

Let $\phi$ be a frozen feature extractor. Move the loss into feature space (paper Eq. 13):

$$\mathcal{L} = \mathbb{E}\Big[\big\|\phi(\mathbf{x}) - \texttt{sg}\big(\phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x}))\big)\big\|^2\Big], \qquad \mathbf{x}=f_\theta(\boldsymbol{\epsilon}), \tag{38}$$

with $\mathbf{V}$ built from $\phi(\mathbf{y}^{+})$ and $\phi(\mathbf{y}^{-})$; and over many features (paper Eq. 14):

$$\mathcal{L} = \sum_j \mathbb{E}\Big[\big\|\phi_j(\mathbf{x}) - \texttt{sg}\big(\phi_j(\mathbf{x}) + \mathbf{V}(\phi_j(\mathbf{x}))\big)\big\|^2\Big]. \tag{39}$$

Gradients flow through $\phi$ (and through a VAE decoder if the generator lives in latent space and $\phi$ in pixel
space). $\phi$ is **not** used at inference.

**This is not a perceptual loss.** A perceptual loss is $\|\phi(\mathbf{x})-\phi(\mathbf{x}_{\text{target}})\|^2$ and
requires a *paired* target. Here the target is $\phi(\mathbf{x})+\mathbf{V}(\phi(\mathbf{x}))$; no pairing exists, and
the objective matches the pushforward distributions $\phi_{\#}q$ and $\phi_{\#}p$.

**It is orthogonal to latent generation.** $\phi$ measures; the generator's output space is a separate choice.

### 9.2 Feature and drift normalisation

For a feature $\phi_j\in\mathbb{R}^{C_j}$, define a scalar scale $S_j$ so that the typical normalised distance is
$\sqrt{C_j}$ (paper Eq. 18–21):

$$\tilde\phi_j = \phi_j/S_j, \qquad S_j = \frac{1}{\sqrt{C_j}}\,\mathbb{E}_{\mathbf{x}}\mathbb{E}_{\mathbf{y}}\big[\|\phi_j(\mathbf{x})-\phi_j(\mathbf{y})\|\big], \tag{40}$$

and use $\tilde\tau_j = \tau\sqrt{C_j}$ in the kernel (paper Eq. 22):

$$k(\mathbf{x},\mathbf{y}) = \exp\!\left(-\frac{1}{\tilde\tau_j}\big\|\tilde\phi_j(\mathbf{x})-\tilde\phi_j(\mathbf{y})\big\|\right). \tag{41}$$

After this, $\tau$ is a pure number, independent of feature magnitude and dimensionality — the paper fixes
$\tau\in\{0.02,0.05,0.2\}$ across all configurations. $S_j$ is computed from the batch with a stop-gradient (it is
conceptually a statistic of the previous batch).

Similarly normalise the drift (paper Eq. 23–25):

$$\tilde{\mathbf{V}}_j = \mathbf{V}_j/\lambda_j, \qquad \lambda_j = \sqrt{\mathbb{E}\Big[\tfrac{1}{C_j}\|\mathbf{V}_j\|^2\Big]}, \tag{42}$$

so the per-feature loss is (paper Eq. 26)

$$\mathcal{L}_j = \mathrm{MSE}\Big(\tilde\phi_j(\mathbf{x}),\; \texttt{sg}\big(\tilde\phi_j(\mathbf{x}) + \tilde{\mathbf{V}}_j\big)\Big), \qquad \mathcal{L} = \sum_j \mathcal{L}_j. \tag{43}$$

Multiple temperatures are aggregated *after* normalisation: $\tilde{\mathbf{V}}_j \leftarrow \sum_\tau \tilde{\mathbf{V}}_{j,\tau}$.

> **Consequence that trips people up.** With drift normalisation, identity (15) still holds — but for a field whose
> magnitude you have deliberately pinned, so $\mathcal{L}\approx\eta^2\cdot\text{const}$ and the training loss stops
> being a convergence diagnostic. Always log the **un-normalised** $\|\mathbf{V}_\tau\|^2$ separately. Notebook 1 §7
> shows the training loss moving 2.2e-2 → 1.3e-2 while the raw drift falls by almost an order of magnitude over the
> same run.

Normalisation scales are shared across spatial locations within a feature map (computing them per location would be
slow and statistically unnecessary).

### 9.3 Why the metric dominates

The field's informativeness depends on the kernel resolving similarity. In a poor metric almost every pair of samples
is far apart, every kernel evaluation sits in the exponential tail, and the drift estimate is *starved* — small,
noisy, and uninformative. This is why the paper's Table 3 shows FID varying by 3.3x from changing only $\phi$:

| encoder $\phi$ | FID (B/2, 100 ep) |
|---|---|
| SimCLR ResNet-50 | 11.05 |
| MoCo-v2 ResNet-50 | 8.41 |
| latent-MAE, width 256, 192 ep | 8.46 |
| latent-MAE, width 640, 192 ep | 6.30 |
| latent-MAE, width 640, 1280 ep | 4.28 |
| latent-MAE, width 640, 1280 ep + classifier ft | **3.36** |

and why the paper reports being unable to train on ImageNet without a feature encoder at all.

Multi-scale, multi-location feature sets help further (paper A.5): per-location vectors, global mean/std, and
mean/std over $2\times2$ and $4\times4$ patches give 9.58 → 9.10 → 8.46 FID. The **std** statistics are what force
matching of local second moments, i.e. real texture rather than a smooth average.

**Restatement of the method:** the generator is the cheap part; the metric is the model.

---

## 10. Classifier-free guidance as a negative mixture

Standard CFG extrapolates model predictions,
$\hat\epsilon = \epsilon_\theta(\cdot|\varnothing) + w(\epsilon_\theta(\cdot|c)-\epsilon_\theta(\cdot|\varnothing))$,
which requires two network evaluations at inference. Drifting achieves the analogous effect by changing **which
negatives it repels from**, entirely at training time.

Define the effective negative distribution (paper Eq. 15)

$$\tilde q(\cdot|c) \;=\; (1-\gamma)\,q_\theta(\cdot|c) + \gamma\,p_{\text{data}}(\cdot|\varnothing), \qquad \gamma\in[0,1). \tag{44}$$

Equilibrium now means $\tilde q(\cdot|c) = p_{\text{data}}(\cdot|c)$. Solving for the model's own distribution (paper Eq. 16):

$$\boxed{\;q_\theta(\cdot|c) = \alpha\,p_{\text{data}}(\cdot|c) - (\alpha-1)\,p_{\text{data}}(\cdot|\varnothing),\qquad \alpha = \frac{1}{1-\gamma}\ge 1\;} \tag{45}$$

The generator is trained to match a **linear extrapolation** of the conditional away from the unconditional — exactly
the spirit of CFG, but as a property of the equilibrium rather than of an inference-time formula.

Implementation (paper A.7): include $N_{\text{unc}}$ real samples from random classes as extra negatives, weighted by
$w$ in the kernel. Then

$$\tilde q(\cdot|c) = \frac{(N_{\text{neg}}-1)\,q_\theta(\cdot|c) + N_{\text{unc}}\,w\,p_{\text{data}}(\cdot|\varnothing)}{(N_{\text{neg}}-1) + N_{\text{unc}}\,w}, \tag{46}$$

$$\gamma = \frac{N_{\text{unc}}w}{(N_{\text{neg}}-1)+N_{\text{unc}}w}, \qquad \alpha = \frac{(N_{\text{neg}}-1)+N_{\text{unc}}w}{N_{\text{neg}}-1}. \tag{47}$$

Given a target $\alpha$, solve (47) for $w$. The same weight $w$ is applied when computing the distance
normalisation.

**CFG-conditioning.** $\alpha$ is sampled per label each step (from e.g. $p(\alpha)\propto\alpha^{-3}$ or
$\alpha^{-5}$ on $[1,4]$), the negatives are built accordingly, and $\alpha$ is fed to the network as a conditioning
input, $f_\theta(\boldsymbol{\epsilon},c,\alpha)$. At inference you pick any $\alpha$ freely — still **1 NFE**, unlike
standard CFG which needs two forward passes per step.

Empirically the FID/IS trade-off mirrors ordinary CFG, and the paper's best model attains its optimal FID at
$\alpha=1.0$ — nominally "no CFG", though training still used unconditional real samples as negatives, so the
mechanism is not idle.

---

## 11. Equivariant drifting on SE(3)

For molecular and structural data the target distribution is invariant under rigid motions, so the field must be
**equivariant**: $\mathbf{V}(R\mathbf{X}+t) = R\,\mathbf{V}(\mathbf{X})$.

The problem is that $\|\mathbf{X}_i-\mathbf{Y}_j\|$ is not rotation-invariant unless the two structures share a frame.
*Drift-React* ([arXiv:2605.22990](https://arxiv.org/abs/2605.22990)) solves it with per-sample Kabsch alignment:

1. Centre and Kabsch-align each generated structure $\mathbf{X}_i$ onto the reference, obtaining rotations $R_i$ and
   aligned copies $\tilde{\mathbf{X}}_i$;
2. build the augmented target set $\mathcal{T}=[\mathbf{Y}^{\star},\tilde{\mathbf{X}}_1,\dots,\tilde{\mathbf{X}}_K]$
   and logits $L_{ij}=-\|\tilde{\mathbf{X}}_i-\mathcal{T}_j\|/\tau$;
3. symmetrise: $A_{ij} = \sqrt{\sigma_{\text{row}}(L)_{ij}\,\sigma_{\text{col}}(L)_{ij}}$;
4. split into attractive $A^{+}_i = A_{i,0}$ and repulsive $A^{-}_{ij}=A_{ij}$, $j\ge1$, and form the field in the
   aligned frame
   $$\tilde{\mathcal{V}}(\tilde{\mathbf{X}}_i) = A^{+}_i\sum_{j=1}^{K} A^{-}_{ij}\,\big(\mathbf{Y}^{\star}-\tilde{\mathbf{X}}_j\big); \tag{48}$$
5. rotate back: $\mathcal{V}(\mathbf{X}_i) = R_i^{\top}\,\tilde{\mathcal{V}}(\tilde{\mathbf{X}}_i)$.

Equivariance holds because the affinities are scalar invariants (functions of aligned pairwise distances) while the
displacement vectors transform covariantly, and $R_i^{\top}$ returns the result to the generator's output frame.
Notebook 8 verifies this numerically: the relative residual is $6\times10^{-15}$ in float64 and $2\times10^{-6}$ in
float32, against $\|\mathbf{V}\|\sim10^{-2}$ — floating-point noise in each case, so the construction is exactly
equivariant rather than approximately so.

Two further ideas from that work generalise well:

* **Single-positive regime.** When each input has exactly one ground-truth output (a reference pathway), naive Monte
  Carlo degenerates: the kernel weight on the lone positive cancels against its own empirical partition function. The
  joint (row+column) normalisation over the augmented set $\mathcal{T}$ fixes this by putting $\mathbf{Y}^{\star}$ and
  the swarm on equal footing. **This is the case that matters for conditional structure prediction**, where the
  dataset is a set of (input, single reference structure) pairs.
* **Hard constraints via output parameterisation.** Boundary conditions are enforced by construction, not by loss:
  multiplying the per-image displacement by a sinusoidal envelope $w(\alpha_k)=\sin(\pi\alpha_k)$ pins the endpoints
  exactly, whatever the network weights.

An alternative to equivariant architectures is an **invariant feature space** — e.g. the vector of all pairwise
interatomic distances $\phi_k(\mathbf{x})=\|x_{i_k}-x_{j_k}\|$, which is translation- and rotation-invariant by
construction, with a sparse analytic Jacobian
($\partial\phi_k/\partial x_a = \pm\hat r_{i_kj_k}$ for $a\in\{i_k,j_k\}$, zero otherwise; 6 nonzeros per row).
Running the drifting loss on $\phi$ makes the field automatically invariant while the generator's output stays in
Cartesian space. Notebook 8 uses this; Notebook 9 shows it is sometimes *better* than working in coordinates.

---

## 12. Score and force guidance: drifting toward Boltzmann

Standard drifting has equilibrium $q=p_{\text{data}}$. In the physical sciences the *desired* target is often
$p_{\text{Boltz}}(\mathbf{x}) \propto \exp(-\mathcal{E}(\mathbf{x})/k_BT)$, and $p_{\text{data}}$ differs from it
because of finite sampling, enhanced-sampling bias, or non-equilibrium data collection.
*Drifting to Boltzmann* ([arXiv:2603.05527](https://arxiv.org/abs/2603.05527)) closes the gap.

### 12.1 The Drifting Score Identity

Using (25) and integration by parts, for a **Gaussian** kernel and any sufficiently regular $p$:

$$\boxed{\;\mathbf{V}^{+}_{p}(\mathbf{x}) = \tau^2\,\mathbb{E}_{p}\big[\bar k(\mathbf{x},\mathbf{y})\,\nabla_{\mathbf{y}}\log p(\mathbf{y})\big]\;} \tag{49}$$

*Proof.* The numerator is $\mathbb{E}_p[k(\mathbf{y}-\mathbf{x})]$. Using
$k(\mathbf{x},\mathbf{y})(\mathbf{y}-\mathbf{x}) = -\tau^2\nabla_{\mathbf{y}}k(\mathbf{x},\mathbf{y})$ and integrating
by parts (boundary terms vanish),
$$\mathbb{E}_p\big[k(\mathbf{y}-\mathbf{x})\big] = -\tau^2\!\int p\,\nabla_{\mathbf{y}}k = \tau^2\!\int (\nabla_{\mathbf{y}}p)\,k = \tau^2\,\mathbb{E}_p\big[k\,\nabla_{\mathbf{y}}\log p\big].$$
Divide by $Z_p=\mathbb{E}_p[k]$. $\square$

*The attraction term is a kernel-weighted average of the score of the sampling distribution.* This is the same
statement as (27), read in the other direction — but now applied to the *sampling* measure rather than to the KDE.

For the Laplace kernel the extra $1/\|\mathbf{y}-\mathbf{x}\|$ spoils it, which is why force-guided drifting
switches to Gaussian.

### 12.2 The Drifting Force Identity

Boltzmann's score is the normalised force, $\nabla\log p_{\text{Boltz}} = \mathbf{F}/k_BT$ with
$\mathbf{F}=-\nabla\mathcal{E}$. Substituting into (49):

$$\mathbf{V}^{+}_{p_{\text{Boltz}}}(\mathbf{x}) = \tau^2\,\mathbb{E}_{p_{\text{Boltz}}}\!\left[\bar k(\mathbf{x},\mathbf{y})\,\frac{\mathbf{F}(\mathbf{y})}{k_BT}\right]. \tag{50}$$

The Boltzmann-weighted attraction is computable **entirely from force labels**, which molecular simulations and neural
network potentials provide.

### 12.3 Force-Interpolated drifting (FI)

Displacement form and force form are two expressions for the same population quantity, so interpolate them per term:

$$\mathbf{V}^{+}_{\omega}(\mathbf{x}) = \sum_j \bar k(\mathbf{x},\mathbf{y}_j^{+})\left[(1-\omega)\big(\mathbf{y}_j^{+}-\mathbf{x}\big) + \omega\,\tau^2\frac{\mathbf{F}(\mathbf{y}_j^{+})}{k_BT}\right]. \tag{51}$$

In the population limit with data from $p_{\text{data}}$, applying (49) to $p_{\text{data}}$ gives

$$\mathbf{V}^{+}_{\omega}(\mathbf{x}) = \mathbf{V}^{+}_{p_{\text{data}}}(\mathbf{x}) + \omega\,\tau^2\,\mathbb{E}_{p_{\text{data}}}\!\left[\bar k(\mathbf{x},\mathbf{y})\,\nabla_{\mathbf{y}}\log\frac{p_{\text{Boltz}}(\mathbf{y})}{p_{\text{data}}(\mathbf{y})}\right]. \tag{52}$$

$\omega$ **linearly scales an explicit Boltzmann correction**: the discrepancy between what the forces say
($\nabla\log p_{\text{Boltz}}$) and what the data says ($\nabla\log p_{\text{data}}$). At $\omega=1$, the full field is

$$\hat{\mathbf{V}}(\mathbf{x}) = \underbrace{\mathbf{V}^{+}_{p_{\text{data}}} - \mathbf{V}^{-}_{q}}_{\text{anti-symmetric drift}} \;+\; \underbrace{\tau^2\,\mathbb{E}_{p_{\text{data}}}\!\left[\bar k\,\nabla\log\tfrac{p_{\text{Boltz}}}{p_{\text{data}}}\right]}_{\text{Boltzmann correction}}. \tag{53}$$

$\omega$ also serves as variance control: in molecular systems
$\|\tau^2\mathbf{F}/k_BT\|$ can be $\sim150\times\|\mathbf{y}-\mathbf{x}\|$, so small $\omega$ (e.g. 0.01) supplies the
full correction *direction* while adding only $O(\omega)$ variance.

### 12.4 Force-Aligned Kernel (FK)

Instead of altering displacements, reweight the kernel toward Boltzmann-likely data:

$$\bar k_{\text{corrected}}(\mathbf{x},\mathbf{y}_j) \propto k(\mathbf{x},\mathbf{y}_j)\,p_{\text{Boltz}}(\mathbf{y}_j)^{\gamma}
\;\Longrightarrow\;
\ell_j = -\frac{\|\mathbf{x}-\mathbf{y}_j\|^2}{2\tau^2} - \frac{\gamma\,\mathcal{E}(\mathbf{y}_j)}{k_BT}. \tag{54}$$

With energy labels this is exact (the $\log Z$ cancels in the softmax). With only forces, Taylor-expand
$\mathcal{E}(\mathbf{y}_j)\approx\mathcal{E}(\mathbf{x}) - \mathbf{F}(\mathbf{y}_j)\!\cdot\!(\mathbf{y}_j-\mathbf{x})$;
$\mathcal{E}(\mathbf{x})$ is constant in $j$ and cancels:

$$\ell_j \approx -\frac{\|\mathbf{x}-\mathbf{y}_j\|^2}{2\tau^2} + \gamma\,\frac{\mathbf{F}(\mathbf{y}_j)\cdot(\mathbf{y}_j-\mathbf{x})}{k_BT}. \tag{55}$$

Geometrically: data points whose force points *away from* the query (so that moving out to $\mathbf{y}_j$ goes
downhill in energy) get upweighted. The approximation error is controlled by
$\varepsilon = \tau\|\nabla^2\mathcal{E}\|/\|\nabla\mathcal{E}\|$ — exact for locally affine potentials, degrading as
$\tau$ grows.

The effective sampling measure becomes $p_{\text{data}}\cdot p_{\text{Boltz}}^{\gamma}/Z$. **A checkable prediction:**
if $p_{\text{data}}$ is itself Boltzmann at a wrong temperature $T_{\text{data}}$, the exact energy-tilted kernel
recovers the target at

$$\gamma^{\star} = \frac{1}{k_BT_{\text{target}}} - \frac{1}{k_BT_{\text{data}}}. \tag{56}$$

Notebook 9 verifies this quantitatively on a 2-D double well: with $T_{\text{data}}=2.5$, $T_{\text{target}}=1.0$,
theory predicts $\gamma^\star=0.6$, and the run at $\gamma=0.6$ reproduces the target's mean energy to 0.3% and both
per-coordinate spreads to under 0.4% — measured against an exactly-computed reference rather than a sampled one,
because an unconverged sampler biases $\langle\mathcal{E}\rangle$ by ~10% on a stiff quartic and would have swamped
the effect. $\gamma=0.2$ under-corrects and $\gamma=1.0$ over-corrects, as predicted.

### 12.5 Equilibrium analysis

For the force-substituted field (53):

* **$q=p_{\text{data}}$ is no longer an equilibrium** when $p_{\text{data}}\ne p_{\text{Boltz}}$: the anti-symmetric part
  cancels but the correction term survives and pushes past $p_{\text{data}}$ toward $p_{\text{Boltz}}$.
* **$q=p_{\text{Boltz}}$ is an exact equilibrium iff $p_{\text{data}}=p_{\text{Boltz}}$.** Otherwise the attraction
  normalises over $p_{\text{data}}$ while the repulsion normalises over $p_{\text{Boltz}}$, the constants differ, and
  cancellation fails.
* **The realised equilibrium lies between the two**, shifted toward Boltzmann by $\omega$ (or $\gamma$).

Force alignment technically breaks strict anti-symmetry, since attraction and repulsion now use different kernels.
That is a *deliberate* shift of the equilibrium, and it is qualitatively different from rescaling the relative
magnitudes of $\mathbf{V}^{+}$ and $\mathbf{V}^{-}$ (§2.3), which collapses the model: here the displacements are
untouched and only the weighting inside attraction changes.

### 12.6 Representation dependence

An empirical finding with no analogue in image generation: **which force-guidance mechanism works reverses with the
representation.**

* In **Cartesian coordinate space**, FI dominates, because forces are literally displacement directions and blending
  them with data displacements is dimensionally and physically natural.
* In **pairwise-distance feature space**, FK dominates, because interpolating a force vector in distance space
  produces an abstract vector *outside the convex hull of valid molecular geometries* — a manifold violation — while
  reweighting the kernel leaves the displacements (differences of real geometries) intact, so bond constraints are
  preserved by construction.

The general principle: **any modification that changes the drift's direction risks leaving the data manifold;
modifications that only change the weights cannot.** Weight-space guidance is the safer default whenever the output
space is constrained. This matters directly for structure prediction, where valid outputs are a thin manifold in
$\mathbb{R}^{3N}$.

The same work also documents a **per-molecule validity crisis**: good aggregate distance-histogram statistics can
coexist with catastrophic per-molecule constraint violations. Aggregate metrics are not enough; always report
per-sample validity.

---

## 13. Discrete and latent drifting

Drifting needs a metric space in which to compute kernels, so discrete data must be embedded. Two patterns.

### 13.1 Latent drifting with a decoder

*Pepti-drift* ([arXiv:2606.27824](https://arxiv.org/abs/2606.27824)) generates peptide sequences by drifting in a
learned, $\ell_2$-normalised latent space and decoding non-autoregressively:

$$u_\theta(y) = \mathrm{norm}\big(\phi_P(e_P(y))\big) \in \mathbb{S}^{d_z-1}, \tag{57}$$

$$z_0 = G_\theta(h_\theta(a), \xi), \qquad z_1 = z_0 + \eta\,D_\theta\big(z_0, h_\theta(a)\big), \qquad \xi\sim\mathcal{N}(0,I), \tag{58}$$

with a frozen protein language model providing $e_P$ and $e_A$. Normalising onto the hypersphere makes cosine
similarity the metric and prevents latents from collapsing toward the origin. Decoding is a single parallel forward
pass, so the whole pipeline is one-step.

Its drift is deliberately **not** anti-symmetric:

$$V^{+}_{\text{att}}(\bar x) = \alpha_{+}\big(u^{+} - \bar x\big), \qquad
V^{-}_{\text{rep}}(\bar x) = \alpha_{-}(e)\sum_{k} w^{-}_{k}\big(\bar x - u^{-}_{k}\big), \tag{59}$$

with $w^{-}_k$ a softmax over distances to the $K$ *hardest* negatives from a fixed pool of toxic/hemolytic peptides.
The target is $\tilde x = \texttt{sg}\big[\mathrm{norm}(\bar x + V)\big]$.

This is **objective-steering rather than distribution-matching**: attraction goes to an antigen-matched positive
teacher, and repulsion goes to an undesirable *region*, not to the model's own samples. So the equilibrium is not
"$q=p$" but "close to binders, far from toxins", and the mixture weight $\alpha_-$ is the knob on that trade-off.
Because the two objectives overlap in latent space, $\alpha_-$ is ramped in via a warm-up schedule

$$\alpha_{-}(e) = \alpha_{-}^{\text{final}}\cdot\min\!\Big(1,\;\frac{\max(0,\,e-e_{\text{start}})}{e_{\text{warmup}}}\Big) \tag{60}$$

so the model first learns a stable binding-oriented trajectory before repulsion introduces competing gradients, and a
displacement penalty $\|\Delta\|^2$ prevents degenerate large jumps.

Notebook 10 reproduces the mechanism on a synthetic motif-grammar task, with one caveat the formulation invites.
Adding a "toxic" motif pool as extra negatives cuts that motif's frequency in generated sequences roughly threefold
while the binder rate *rises* — the two compete for positions, so repelling from one selects for the other. But the
field as written has **no anti-symmetric self-repulsion**, and on that task it collapses to about eight distinct
sequences out of four thousand: with a uniformly-random positive teacher the MSE minimiser is provably the
per-antigen mean latent, and with a kernel-weighted teacher nothing spreads the samples far enough apart for the
kernel to distinguish them. Restoring the model's own samples in the negative pool fixes it, and the two repulsion
terms then trade off along near-orthogonal axes (safety vs affinity, diversity vs affinity). The warm-up schedule
(60) made no measurable difference in that setting, which is consistent with its stated rationale: the toxic pool
sat far enough away in latent space that repulsion was weak early anyway.

### 13.2 Drifting from a learned prior

*Drift-AR* ([arXiv:2603.28049](https://arxiv.org/abs/2603.28049)) replaces the Gaussian prior with a **model-dependent**
one: an autoregressive backbone predicts per-position features, and the prior variance is tied to the AR model's
per-position entropy,

$$q\big(x^{(r)}\mid z^{(r)}_{\text{AR}}, \mathcal{E}^{(r)}\big) = \mathcal{N}\!\Big(z^{(r)}_{\text{AR}},\;\sigma^2(\mathcal{E}^{(r)})\,I\Big),
\qquad \sigma(\mathcal{E}) = \sigma_{\max}\frac{e^{\mathcal{E}/\tau_\sigma}-1}{e^{\mathcal{E}_{\max}/\tau_\sigma}-1}. \tag{61}$$

Confident positions get near-zero variance (so the field naturally vanishes there); uncertain positions get broad
priors and strong drift. The drifting loss is then computed in a frozen feature space as usual.

The transferable lesson is the failure mode they name: the **moving prior** problem. If the prior is produced by a
module that is itself being updated, $q$ changes every iteration and the field chases a target that will not hold
still. Their fix is annealed/staged training — freeze or slow the prior-producing module while the drifting decoder
stabilises. Any attempt to drift *inside* a larger trainable system (a structure-prediction trunk, for example) must
confront this.

---

## 14. Unified comparison with other paradigms

| | inference cost | training signal | time variable | mode collapse risk | adversarial |
|---|---|---|---|---|---|
| **DDPM / score SDE** | $N$ NFE (50–1000) | denoising regression at sampled $t$ | yes | low | no |
| **Flow matching** | $N$ NFE (ODE solve) | velocity regression on interpolants | yes | low | no |
| **Consistency / iCT** | 1–4 NFE | self-consistency across adjacent $t$ | yes | low | no |
| **MeanFlow / iMeanFlow** | 1 NFE | average-velocity identity | yes | low | no |
| **Distillation** | 1 NFE | match a pretrained teacher | inherited | low | sometimes |
| **GAN** | 1 NFE | discriminator gradient | no | **high** | **yes** |
| **MMD / moment matching** | 1 NFE | gradient of a kernel discrepancy | no | medium | no |
| **Normalising flow** | 1 NFE (inverse) | exact log-likelihood | no | low | no |
| **VAE** | 1 NFE | ELBO | no | medium (posterior collapse) | no |
| **Drifting** | **1 NFE** | regression onto a drifted sample | **no** | low (repulsion is built in) | no |

Drifting occupies a specific corner: **one-step, non-adversarial, no ODE/SDE, no time variable, trained from
scratch.** Every other 1-NFE method either distils a multi-step teacher, retains a time-indexed velocity field, or
uses a discriminator.

Reported results on ImageNet 256×256, 50K images, all trained from scratch:

| method | space | params | NFE | FID |
|---|---|---|---|---|
| DiT-XL/2 | SD-VAE | 675M+49M | 250×2 | 2.27 |
| SiT-XL/2 + REPA | SD-VAE | 675M+49M | 250×2 | 1.42 |
| iCT-XL/2 | SD-VAE | 675M | 1 | 34.24 |
| Shortcut-XL/2 | SD-VAE | 675M | 1 | 10.60 |
| MeanFlow-XL/2 | SD-VAE | 676M | 1 | 3.43 |
| iMeanFlow-XL/2 | SD-VAE | 610M | 1 | 1.72 |
| **Drifting, B/2** | SD-VAE | **133M** | 1 | 1.75 |
| **Drifting, L/2** | SD-VAE | 463M | 1 | **1.54** |
| StyleGAN-XL (pixel) | pix | 166M | 1 | 2.30 |
| **Drifting, L/16 (pixel)** | pix | 464M | 1 | **1.61** |

The 133M-parameter Base model matching 610–676M one-step XL models is the headline efficiency result; the pixel-space
number at 87 GFLOPs versus StyleGAN-XL's 1574 GFLOPs is the headline compute result.

Beyond images, the paper's *Drifting Policy* replaces the multi-step generator in Diffusion Policy with a 1-NFE
drifting model, computing the drift on raw control representations with **no feature space**, and matches or beats
100-NFE Diffusion Policy across Robomimic tasks (notably BlockPush phase 1: 0.36 → 0.56).

---

## 15. Open problems

Stated by the authors, plus what the follow-up literature has and has not settled.

1. **The converse of the equilibrium theorem.** $q=p\Rightarrow\mathbf{V}=\mathbf{0}$ is a two-line proof; the useful
   direction has only the bilinear heuristic of §6 with a non-degeneracy assumption that is hard to check. What are
   verifiable conditions on $(k,\tau,\phi)$ under which $\|\mathbf{V}\|\le\varepsilon$ bounds a real divergence?

2. **The cost of non-conservativeness.** The Laplace field is not a gradient field (§7.3). The finite-particle
   analysis (§7.4) shows this introduces an irreducible scale-mismatch residual. Is there a normalised, conservative
   field that keeps the Laplace kernel's dynamic range? "Sharp-normalised" constructions are a candidate.

3. **Kernel and field design.** The authors explicitly flag their instantiation as possibly sub-optimal. The whole
   class (5) is unexplored: learned interaction kernels, anisotropic metrics, higher-order interactions.

4. **Eliminating the feature encoder.** The dependence on a pretrained SSL encoder is the method's biggest practical
   liability — it is an extra model, an extra training stage, and a domain-specific asset. Can the metric be learned
   jointly without reintroducing adversarial dynamics or the moving-prior problem (§13.2)?

5. **Scaling behaviour.** Drifting requires large batches for a low-variance field estimate, so its scaling laws in
   batch size, positives-per-class, and model size may differ qualitatively from diffusion's. Largely uncharted.

6. **Diversity, coverage, and memorisation.** The repulsion term is an explicit diversity pressure, which is
   attractive, but small $\tau$ makes the field nearly "copy your nearest neighbour". The paper's nearest-neighbour
   analysis finds no memorisation, but there is no theory bounding it as a function of $\tau$ and $N_{\text{pos}}$.

7. **Genuinely conditional generation with one reference.** Most scientific datasets are (input, single reference
   output) pairs. Drift-React's joint normalisation over an augmented target set (§11) is the current answer; whether
   it recovers *aleatoric* diversity — the ensemble of plausible outputs for one input — rather than just the mean, is
   open, and it is the crux of applying drifting to structure prediction (see `PROTENIX_DRIFTING.md`).

---

## Appendix: symbol table

| symbol | meaning |
|---|---|
| $p_{\boldsymbol{\epsilon}}$, $\boldsymbol{\epsilon}$ | prior distribution and a draw from it |
| $f_\theta$, $q_\theta$ | generator, and its pushforward $[f_\theta]_{\#}p_{\boldsymbol{\epsilon}}$ |
| $p$, $p_{\text{data}}$ | target distribution |
| $\mathbf{y}^{+}$, $\mathbf{y}^{-}$ | positive (from $p$) and negative (from $q$) samples |
| $k$, $\tilde k$ | kernel and its normalised form $k/Z$ |
| $\tau$ | kernel temperature (dimensionless after feature normalisation) |
| $\mathbf{V}^{+}_{p}$, $\mathbf{V}^{-}_{q}$ | attractive and repulsive mean-shift fields |
| $\mathbf{V}_{p,q}$ | drifting field, $\mathbf{V}^{+}_{p}-\mathbf{V}^{-}_{q}$ |
| $\eta$ | explicit drift step size |
| $\phi$, $\phi_j$ | feature extractor; its $j$-th feature |
| $S_j$, $\lambda_j$ | feature and drift normalisation scales |
| $\alpha$, $\gamma$, $w$ | CFG strength, negative-mixture rate, kernel weight on unconditional negatives |
| $\omega$ | force-interpolation weight (FI) |
| $\gamma$ (§12.4) | energy-tilt exponent (FK) |
| $\texttt{sg}$ | stop-gradient |
| $N_c, N_{\text{pos}}, N_{\text{neg}}, N_{\text{unc}}$ | classes per batch; positives, negatives, unconditional negatives per class |
