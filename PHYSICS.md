# The Physics of Drifting Models

`THEORY.md` derives drifting as mathematics: a field, a fixed point, a loss. This document reads the same objects as
**physics**, because almost every design decision in the method has a physical twin, and the twin usually tells you
what will go wrong before the experiment does.

The central claim is a single sentence:

> **Diffusion is a story about a stochastic process; drifting is a story about an interacting particle system finding
> mechanical equilibrium.**

That difference — process versus equilibrium, thermal versus athermal, time-indexed versus timeless — organises
everything below.

## Table of Contents

1. [Prologue: two different physics](#1-prologue-two-different-physics)
2. [The two-component plasma](#2-the-two-component-plasma)
3. [Mean shift as a force from a smeared density](#3-mean-shift-as-a-force-from-a-smeared-density)
4. [Nonreciprocity and Newton's third law for distributions](#4-nonreciprocity-and-newtons-third-law-for-distributions)
5. [Mechanical equilibrium, and where the temperature isn't](#5-mechanical-equilibrium-and-where-the-temperature-isnt)
6. [Overdamped dynamics, mobility, and the implicit step size](#6-overdamped-dynamics-mobility-and-the-implicit-step-size)
7. [Screening, correlation functions, and the inverse problem](#7-screening-correlation-functions-and-the-inverse-problem)
8. [Force matching and coarse-graining](#8-force-matching-and-coarse-graining)
9. [Symmetry, gauge fixing, and equivariance](#9-symmetry-gauge-fixing-and-equivariance)
10. [Non-conservative fields and the impossibility of an energy](#10-non-conservative-fields-and-the-impossibility-of-an-energy)
11. [Temperature as resolution: a renormalisation reading](#11-temperature-as-resolution-a-renormalisation-reading)
12. [Finite-particle effects and the N-body problem](#12-finite-particle-effects-and-the-n-body-problem)
13. [Where the analogy misleads](#13-where-the-analogy-misleads)
14. [Dictionary](#14-dictionary)

---

## 1. Prologue: two different physics

The physics of diffusion models (see the companion `PHYSICS.md` in `learning-diffusion`) is the physics of the
Langevin equation. A particle is kicked by a thermal bath, its density obeys Fokker–Planck, the reverse-time process
exists by Anderson's theorem, and the score $\nabla\log p$ is a thermodynamic force. Everything is indexed by time,
and time is real: the noise schedule is a temperature protocol, quasi-static annealing gives better samples, and the
number of steps is a physical quantity.

Drifting has none of that. There is no bath, no noise, no time variable, and no reverse process. What there is:

* a **fixed, quenched** set of charges — the data samples $\mathbf{y}^{+}$;
* a **mobile** set of like charges — the model's samples $\mathbf{y}^{-} \sim q_\theta$;
* an **interaction** that attracts the mobile charges to the fixed ones and repels them from each other;
* **overdamped, athermal dynamics**: displacement proportional to net force, no inertia, no fluctuation;
* and an **equilibrium condition** that is *mechanical* — every particle feels zero net force — not thermal.

| | diffusion | drifting |
|---|---|---|
| governing equation | Fokker–Planck / Langevin SDE | interacting-particle ODE (McKean–Vlasov) |
| stochasticity | essential (thermal bath) | absent (deterministic field) |
| time | a real, physical coordinate | none |
| equilibrium | Boltzmann distribution at $T$ | force balance, $\mathbf{V}=\mathbf{0}$ |
| what the model learns | a force field indexed by temperature | a map whose image is force-balanced |
| what makes it stable | detailed balance | anti-symmetry |
| the analogue of entropy | thermal noise | repulsion |
| the analogue of temperature | the noise level $\sigma_t$ | *nothing* — $\tau$ is a length, see §5 |

The last two rows are the ones people get wrong, so they get their own sections.

---

## 2. The two-component plasma

### 2.1 The setup

Write the drifting field in its mean-shift form (`THEORY.md` §3.2):

$$\mathbf{V}(\mathbf{x}) \;=\; \underbrace{\sum_j \bar k(\mathbf{x},\mathbf{y}^{+}_j)\,(\mathbf{y}^{+}_j-\mathbf{x})}_{\mathbf{V}^{+}}
\;-\; \underbrace{\sum_j \bar k(\mathbf{x},\mathbf{y}^{-}_j)\,(\mathbf{y}^{-}_j-\mathbf{x})}_{\mathbf{V}^{-}}$$

This is exactly the force on a test charge in a **two-component system**:

* the positives $\{\mathbf{y}^{+}\}$ are **quenched** charges of one sign — fixed in place, not free to move, drawn
  once from $p_{\text{data}}$;
* the negatives $\{\mathbf{y}^{-}\}$ are **mobile** charges of the opposite sign to the test charge's attraction,
  i.e. the same sign as the test charge, so they push it away;
* the test charge is itself one of the mobile particles.

The physical system this most resembles is a **two-component plasma** or a Coulomb gas with a neutralising
background, and the phenomenon that matters is **screening**.

### 2.2 Screening is what equilibrium means

In an electrolyte, a fixed charge is surrounded by a cloud of mobile counter-charges that cancels its field beyond a
characteristic distance — the Debye length $\lambda_D$. The system reaches equilibrium when every fixed charge is
*perfectly screened*: a test charge placed anywhere feels nothing, because attraction to the fixed charges is exactly
cancelled by repulsion from the mobile cloud.

That is precisely the drifting equilibrium. $\mathbf{V}(\mathbf{x})=\mathbf{0}$ for all $\mathbf{x}$ says: the mobile
distribution has arranged itself so that its repulsion cancels the data's attraction, everywhere. And the condition
under which this happens is that the mobile density *equals* the fixed density, $q=p$ — perfect screening requires
the screening cloud to have the same shape as what it screens (`THEORY.md` §2.2).

This is a genuinely useful intuition. It says:

* **Under-screening** (too little mobile density somewhere) leaves a net attractive field there, which pulls samples
  in. That is mode-filling.
* **Over-screening** (too much) leaves a net repulsive field, pushing samples out. That is the anti-collapse
  mechanism, and it is why drifting does not need a discriminator to notice a dropped mode: an unscreened region
  *automatically* generates the force that fills it.
* **The field's magnitude is a local measure of screening failure.** Logging $\|\mathbf{V}\|^2$ (Notebook 3 §7) is
  logging how far from neutrality you are.

### 2.3 But the interaction is short-ranged

The Coulomb analogy has one important defect: real Coulomb interactions are long-ranged ($1/r$ in 3-D), which is what
makes screening a nontrivial collective phenomenon with a characteristic length. Drifting's kernel is
$\exp(-\|\mathbf{x}-\mathbf{y}\|/\tau)$ — a **Yukawa** or **screened Coulomb** form, already short-ranged with range
$\tau$.

So drifting is closer to a system of particles with **finite-range interactions**, and $\tau$ plays the role that
$\lambda_D$ plays in a plasma: it is the distance beyond which a particle does not feel another. The differences this
makes:

* **No long-range collective modes.** A plasma has plasmon oscillations because the interaction is long-ranged.
  Drifting's dynamics has no such collective excitations; disturbances do not propagate beyond $\tau$.
* **The field can be *starved*.** If every pair is separated by $\gg\tau$, the raw kernel weights all underflow and
  the field carries no information. This is Notebook 3 §8's "starved kernel" and it has no plasma analogue — a
  Coulomb system always feels something.
* **Softmax normalisation partly restores long range.** Dividing by the partition function means a lone far-away
  particle is still attracted to the *nearest* data (Notebook 6 §3): direction survives even when magnitude would
  have vanished. There is no physical interaction that does this — it is a deliberately unphysical fix, and it is
  the single most important departure from the plasma picture.

That last point deserves emphasis. **The normalisation is what makes drifting work and what makes it unphysical.**
Every real pair interaction is a function of the pair alone; drifting's effective interaction between $\mathbf{x}$
and $\mathbf{y}_j$ depends on *where all the other particles are*, through $Z(\mathbf{x}) = \sum_{j'} k(\mathbf{x},\mathbf{y}_{j'})$.
Physics has a name for this class — **many-body, density-dependent potentials** — and they arise in coarse-grained
models where degrees of freedom have been integrated out (§7.3). They are known to be badly behaved in exactly the
way one would expect: they are not transferable between state points, and the "potential" changes when you change
the density.

---

## 3. Mean shift as a force from a smeared density

### 3.1 The Gaussian case: an exact potential

Take the Gaussian kernel $k(\mathbf{x},\mathbf{y})=\exp(-\|\mathbf{x}-\mathbf{y}\|^2/2\tau^2)$ and define the kernel
density estimate $\hat p_\tau(\mathbf{x}) = \frac{1}{N}\sum_j k(\mathbf{x},\mathbf{y}_j)$ — physically, the data's
charge density **smeared** on the scale $\tau$, exactly as one smears point charges into a continuous distribution
before solving Poisson's equation.

Then (`THEORY.md` §7.1, verified numerically in Notebook 0a §6)

$$\mathbf{V}^{+}(\mathbf{x}) = \tau^2\,\nabla_{\mathbf{x}}\log \hat p_\tau(\mathbf{x}).$$

So the Gaussian attraction term is *exactly* the gradient of a potential

$$\Phi^{+}(\mathbf{x}) = -\tau^2\log\hat p_\tau(\mathbf{x}),$$

which is the **potential of mean force** of the smeared data density, with $k_BT$ set to $\tau^2$. That is a real
correspondence and it is worth stating carefully: for the Gaussian kernel, drifting's attraction is a conservative
force derived from a free-energy-like potential $\Phi = -k_BT\log\rho$, the standard relation between a density and
its associated potential of mean force in statistical mechanics.

The full field, attraction minus repulsion, is then

$$\mathbf{V} = -\nabla\big[\Phi^{+} - \Phi^{-}\big] = \tau^2\,\nabla\log\frac{\hat p_\tau}{\hat q_\tau},$$

the gradient of a smeared **log-likelihood ratio**. Equilibrium $\mathbf{V}=0$ means the ratio is constant, hence
$\hat q_\tau = \hat p_\tau$: the two smeared densities coincide at resolution $\tau$. This is the cleanest statement
of what drifting does, and the reason summing over several $\tau$ matters (§11).

### 3.2 The Laplace case: no potential exists

The paper's default kernel is Laplace, $k=\exp(-\|\mathbf{x}-\mathbf{y}\|/\tau)$, and for it the identity above
**fails**. Working through the gradient,

$$\nabla_{\mathbf{x}}\log\hat p^{\text{Lap}}_\tau(\mathbf{x}) = \frac{1}{\tau}\sum_j \bar k(\mathbf{x},\mathbf{y}_j)\,\frac{\mathbf{y}_j-\mathbf{x}}{\|\mathbf{y}_j-\mathbf{x}\|},$$

which weights each displacement by $1/\|\mathbf{y}_j-\mathbf{x}\|$. The Laplace mean-shift vector has no such factor.
The two differ, and the difference is not a constant multiple, so **the Laplace field is not the gradient of
anything** (`THEORY.md` §7.3).

Physically: there is no potential energy function whose descent the dynamics performs. There is no conserved
quantity, no Lyapunov function derived from an energy, no Boltzmann weight $e^{-\Phi/k_BT}$ associated with the
steady state. §10 works out what follows.

Why then use it? Because the $1/r$ weighting is exactly what one wants numerically. In high dimensions, distances
concentrate: nearly all pairs sit at nearly the same distance, so a Gaussian kernel's $e^{-r^2}$ produces weights
spanning many orders of magnitude within a narrow band of $r$, and the field becomes "copy the single nearest
neighbour". The Laplace kernel's $e^{-r}$ has a far gentler profile and keeps many neighbours in play — measurable
directly as the effective sample size of the kernel weights (Notebook 3 §7b). **Drifting trades away the existence of
a potential for a usable dynamic range in high dimensions.** That is the trade, stated plainly, and Notebook 9 shows
the trade running the other way when force labels are available: force guidance *needs* the Gaussian, because the
score identity is exactly the statement of §3.1.

---

## 4. Nonreciprocity and Newton's third law for distributions

### 4.1 The pair interaction is nonreciprocal

In any closed physical system, $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$: the force particle $i$ exerts on $j$ is equal
and opposite to the force $j$ exerts on $i$. This is Newton's third law, and it is what guarantees momentum
conservation.

Drifting's effective pair interaction violates it. The force on $\mathbf{x}_i$ from $\mathbf{y}_j$ is
$\bar k(\mathbf{x}_i,\mathbf{y}_j)(\mathbf{y}_j-\mathbf{x}_i)$ with
$\bar k = k/Z(\mathbf{x}_i)$ — normalised by $\mathbf{x}_i$'s **own** partition function. The reciprocal force would
carry $Z(\mathbf{y}_j)$ instead, and these differ. Even setting aside that the positives are quenched (so they do not
move at all), the interaction is asymmetric in the pair.

Nonreciprocal interactions are a live area of physics — active matter, chemically-driven colloids, predator–prey
systems, non-Hermitian dynamics — and they are known to produce behaviour that reciprocal systems cannot:

* **No free energy** to minimise, hence no equilibrium statistical mechanics;
* **Travelling and oscillating steady states**, where the system settles into persistent motion rather than a fixed
  configuration;
* **Exceptional points** in the linearised dynamics, where eigenvectors coalesce and the response becomes singular.

This is not decoration. It is the physical reason the convergence analysis of `THEORY.md` §7.4 has to work harder for
the non-conservative case, and the reason a residual term appears there that has no analogue in gradient-flow theory.

### 4.2 But there *is* reciprocity at the level of distributions

Here is the beautiful part. Although the pair interaction is nonreciprocal, the *functional* is antisymmetric
(`THEORY.md` §2.1):

$$\mathbf{V}_{p,q} = -\mathbf{V}_{q,p}.$$

Swap the roles of "data" and "model" and the field flips sign exactly. This is **Newton's third law promoted from
pairs to distributions**, and it plays exactly the role the microscopic version plays in physics: it is the
structural identity that guarantees the existence and stability of the equilibrium.

Notebook 1 §9 reproduces the paper's ablation showing that breaking it — by rescaling $\mathbf{V}^{+}$ relative to
$\mathbf{V}^{-}$, i.e. by making attraction and repulsion unequal — collapses the model. That is the generative-model
version of what happens to a mechanical system when action and reaction are unbalanced: it accelerates away.

The precision matters. Two distinct things can be broken:

| what is broken | physical analogue | consequence |
|---|---|---|
| pairwise reciprocity ($\bar k$ normalisation) | nonreciprocal interaction, active matter | no potential, no free energy, harder analysis — but a fixed point still exists |
| distributional anti-symmetry ($\alpha_+ \neq \alpha_-$) | unbalanced action/reaction | **no fixed point at $q=p$**; collapse or explosion |

Drifting breaks the first deliberately (it is what normalisation costs) and preserves the second absolutely. Force
guidance (`THEORY.md` §12.5, Notebook 9 §5) breaks the second deliberately too — and the equilibrium *moves* rather
than vanishing, because the displacements are untouched and only the weights change. The distinction between
"changing directions" and "changing weights" that Notebook 9 §6 establishes empirically is, in physical terms, the
distinction between breaking the force law and re-weighting the ensemble.

---

## 5. Mechanical equilibrium, and where the temperature isn't

### 5.1 $\tau$ is a length, not a temperature

The softmax weight $\bar k = e^{-d/\tau}/Z$ looks exactly like a Boltzmann factor $e^{-E/k_BT}/Z$, and this invites a
mistake worth naming explicitly:

> **$\tau$ is not a temperature. It is a screening length / kernel bandwidth. The resemblance is formal.**

The differences are structural, not cosmetic:

* A Boltzmann factor weights **states by energy**; the drifting softmax weights **neighbours by distance**. Distance
  is not an energy; there is no Hamiltonian here.
* Temperature is conjugate to energy and controls **fluctuations**. There are no fluctuations in drifting's dynamics
  — the field is deterministic. Changing $\tau$ changes *which neighbours are visible*, not how noisy the trajectory
  is.
* Temperature has an operational meaning through the fluctuation–dissipation theorem: $D = \mu k_BT$ relates the
  diffusion constant to the mobility. Drifting has $D=0$ identically, so this relation gives $T=0$: **drifting's
  dynamics is athermal, at zero temperature**, whatever $\tau$ is set to.

The one place a genuine temperature appears is force guidance (`THEORY.md` §12, Notebook 9), where $k_BT$ enters
because the *target* is a Boltzmann distribution. There $T$ is the physical temperature of the system being modelled,
completely unrelated to $\tau$. In the notation of Notebook 9, $\tau=0.35$ and $T_{\text{target}}=1.0$ are quantities
of different kinds that happen to be numbers.

### 5.2 Repulsion is the entropic force

If the dynamics is athermal, what stops every sample collapsing onto the nearest data point? In a thermal system,
entropy does: the free energy $F = U - TS$ is minimised at finite spread because the entropy term rewards
delocalisation.

In drifting, the repulsion term does the same job by a completely different mechanism. It is an **explicit entropic
force**, in the same sense as the entropic forces of soft-matter physics:

* **Excluded-volume / steric repulsion.** Hard spheres have no energetic interaction at all; their effective
  repulsion is purely entropic, arising because configurations where they overlap are simply unavailable. Drifting's
  $\mathbf{V}^{-}$ is the same kind of object: it does not encode any energy, it encodes "you are too close to
  another sample".
* **Depletion forces**, where the density of one species generates an effective potential on another.
* **Entropic elasticity** of a polymer, where the restoring force comes from configuration counting.

This gives the right intuition for the failure modes. In a thermal system you raise $T$ to prevent collapse; in
drifting you have no such knob, so **collapse must be prevented structurally**, and it is: the repulsion is present
from step 0 and cannot be defeated (Notebook 6 §4). Conversely, when repulsion is *removed* — as in Pepti-drift,
where negatives are a fixed external pool rather than the model's own samples — nothing prevents collapse, and
Notebook 10 §5 measures it happening: about eight distinct peptides out of four thousand.

The equivalence is quantitative in the Gaussian case. From §3.1, the full field is
$\tau^2\nabla\log(\hat p_\tau/\hat q_\tau)$; the $-\log\hat q_\tau$ half is precisely $+\tau^2\times$ the gradient of
the *self-entropy* of the sample ensemble at resolution $\tau$. Maximising the samples' own differential entropy at
scale $\tau$ **is** the repulsion term. Drifting is minimising a smeared cross-entropy while maximising a smeared
self-entropy — free-energy minimisation with $\tau^2$ in the role of $k_BT$, for the Gaussian kernel only.

### 5.3 Mechanical, not thermal, equilibrium

Two distinct notions of equilibrium in physics:

* **Thermal equilibrium**: the ensemble is stationary under the dynamics and satisfies detailed balance; the
  distribution is Boltzmann. Diffusion models target this.
* **Mechanical equilibrium**: every particle feels zero net force; the configuration is stationary. A crystal at
  $T=0$, a soap film, a hanging chain.

Drifting targets the second. $\mathbf{V}=\mathbf{0}$ is a force-balance condition, not a stationarity-plus-detailed-balance
condition. This has consequences:

* **No detailed balance, and none needed.** There is no transition kernel whose reversibility one could check.
* **Uniqueness is not guaranteed by the usual arguments.** A thermal system has a unique Boltzmann state under mild
  conditions; a mechanical system can have many force-balanced configurations (think of the many stable packings of
  a granular material). This is precisely the identifiability question of `THEORY.md` §6 in physical dress: does
  force balance pin down the configuration? The honest answer there — a bilinear argument with an unverifiable
  non-degeneracy assumption — is the same honest answer physics gives about jammed states.
* **Escape from a bad configuration cannot rely on thermal activation.** In Langevin sampling you cross barriers by
  waiting; in drifting there is no noise to wait for, so every rearrangement must be driven by the field itself.
  This is why initialisation robustness (Notebook 1 §10) is a nontrivial property rather than an obvious one — an
  athermal system that starts collapsed has only repulsion to un-collapse it.

---

## 6. Overdamped dynamics, mobility, and the implicit step size

### 6.1 Aristotle, not Newton

The drifting update is

$$\mathbf{x} \leftarrow \mathbf{x} + \eta\,\mathbf{V}(\mathbf{x}),$$

i.e. **displacement proportional to force**. There is no inertia, no momentum, no acceleration. In mechanics this is
the **overdamped** or **Aristotelian** limit: the regime where friction dominates so completely that the velocity
instantaneously equilibrates to the applied force,

$$\gamma\dot{\mathbf{x}} = \mathbf{F} \quad\Longrightarrow\quad \dot{\mathbf{x}} = \mu\mathbf{F}, \qquad \mu = 1/\gamma.$$

Everything in generative modelling is overdamped, for a good reason: momentum would make the dynamics
history-dependent, and the object being modelled — a distribution — has no history. (Momentum-based *samplers* like
HMC do exist, but they use momentum as an auxiliary variable to be integrated out, not as physics.)

The step size $\eta$ is therefore a **mobility**: it converts force to displacement, and its units are
(length)/(force). This is the right way to think about why $\eta$ needs the drift normalisation of `THEORY.md` §9.2:
without it, the "force" has whatever units the feature space happens to have, and the mobility must be retuned for
every space. Normalising $\mathbf{V}$ to unit RMS fixes the force scale, so $\eta$ becomes a pure length in units of
the typical inter-sample distance.

### 6.2 The implicit step size, and why the loss lies

Drifting does not actually move samples. It regresses the *model* onto a moved target:

$$\mathcal{L}(\theta) = \big\|f_\theta(\boldsymbol\epsilon) - \mathtt{sg}\big[f_\theta(\boldsymbol\epsilon) + \eta\mathbf{V}\big]\big\|^2.$$

A gradient step on this moves the model's output only a *fraction* of the way to the target — the fraction set by the
learning rate (`THEORY.md` §4.5, Notebook 0c §4). So the realised mobility is

$$\mu_{\text{eff}} \;\propto\; \eta \times \text{lr},$$

which is why Notebook 3 §4 finds a *diagonal valley* in the $(\eta,\text{lr})$ plane rather than two independent
optima: only the product is physical. Physically this is a two-stage friction — the field proposes a displacement,
the optimiser applies a damped version of it — and only the composition has meaning.

The same structure explains why the training loss stops being informative once you normalise the drift. With
$\|\mathbf{V}\|$ pinned to a constant, $\mathcal{L}\approx\eta^2\times\text{const}$ regardless of how far you are
from equilibrium: you are measuring the *step size*, not the *residual force*. The physical quantity is the
un-normalised $\|\mathbf{V}\|^2$, which is the net force per particle and does go to zero. Notebook 3 §7 makes this a
rule; §2.2 above explains why it is the right rule.

### 6.3 The continuum limit: Wasserstein gradient flow

Take $\eta\to0$ and $N\to\infty$. The particle dynamics $\dot{\mathbf{x}}=\mathbf{V}(\mathbf{x})$ with a field that
depends on the particles' own distribution is a **McKean–Vlasov** equation, and the induced evolution of the density
obeys the continuity equation

$$\partial_t q_t + \nabla\!\cdot\!(q_t\mathbf{V}) = 0.$$

For the Gaussian kernel, where $\mathbf{V}=-\nabla\Phi$ with $\Phi = \tau^2\log(\hat q_\tau/\hat p_\tau)$, this is a
**Wasserstein-2 gradient flow** of a smeared KL-like functional (`THEORY.md` §7.2, Notebook 0b §4). The physics: this
is the same variational structure that underlies the JKO scheme, and it is the reason gradient-flow intuitions —
monotone decrease of the functional, convergence to the minimiser — apply at all.

For the Laplace kernel there is no such functional (§3.2), and one is left with a continuity equation driven by a
non-conservative field. Convergence must then be proved directly rather than via a Lyapunov functional, which is
exactly the shape of the analysis in `THEORY.md` §7.4 and the origin of its extra scale-mismatch residual.

---

## 7. Screening, correlation functions, and the inverse problem

### 7.1 What $\tau$ screens

In Debye–Hückel theory, a charge in an electrolyte is screened over the length

$$\lambda_D = \sqrt{\frac{\varepsilon k_BT}{\sum_i n_i q_i^2}},$$

which shrinks as the density of mobile charges grows. Drifting's $\tau$ is externally imposed rather than emergent,
but it plays the same role: beyond $\tau$, a sample does not feel another sample.

The practical consequence is the one Notebook 3 §7 tracks as **effective sample size**,
$\mathrm{ESS}=1/\sum_j\bar k_j^2$: the number of neighbours actually inside the screening radius. In a plasma the
analogous quantity is the number of particles in a Debye sphere, $N_D = \tfrac{4}{3}\pi\lambda_D^3 n$, and it plays
exactly the same diagnostic role:

* $N_D \gg 1$ — the **collective** regime, where many particles contribute and mean-field theory is accurate;
* $N_D \sim 1$ — the **strongly-coupled** regime, where individual pair encounters dominate and mean-field theory
  fails.

$\mathrm{ESS}\approx1$ in drifting is the strongly-coupled regime: the field is "copy your nearest neighbour", which
is both a poor estimator and a memorisation risk. $\mathrm{ESS}\approx N$ is the opposite failure — the field sees
only the mean, and matches moments rather than the distribution. **The plasma parameter and the kernel ESS are the
same diagnostic**, and the healthy range is the same: comfortably many, but far from all.

### 7.2 Correlation functions

The natural language for "does $q$ equal $p$" in liquid-state physics is the **radial distribution function** $g(r)$
and its Fourier transform the **structure factor** $S(k)$. Two ensembles with the same $g(r)$ agree on all pair
correlations, which is much weaker than agreeing as distributions — but is exactly what any pair-interaction-based
method can see.

Drifting is a pair-interaction method. Its field is built from pairwise distances only, so what it can distinguish is
bounded by what pair correlations distinguish. This is the physical statement of the identifiability gap in
`THEORY.md` §6 and Notebook 2 §7: two different distributions with identical pair statistics at every $\tau$ used
would generate the same field.

In practice the gap is smaller than this argument suggests, because the *normalised* kernel is not a pair
interaction — the normalisation makes the effective weight depend on all other particles (§2.3), so the field carries
some higher-order information. That is a genuine advantage of the normalisation beyond numerics, and to our knowledge
it has not been quantified.

### 7.3 Henderson's theorem and its failure

Liquid-state physics has a uniqueness theorem that is the exact twin of drifting's open problem.

> **Henderson's theorem (1974).** For a classical fluid at fixed density and temperature, two pair potentials that
> produce the same radial distribution function differ by at most a constant.

Read as an inverse problem: the pair correlation function *determines* the pair potential. If the analogy were exact,
it would say that matching the field determines the distribution — the converse that `THEORY.md` §6 wants and does
not have.

Why the analogy is not exact, and the failure is instructive:

1. **Henderson's theorem is about a potential, given a distribution.** Drifting asks the reverse: given that the
   field (built from the distribution) vanishes, is the distribution determined? These are adjoint questions, not
   the same one.
2. **Henderson requires a genuine pair potential.** Drifting's normalised interaction is density-dependent, and for
   density-dependent potentials Henderson's uniqueness is known to fail — this is the *representability problem* of
   coarse-graining, where a coarse-grained model reproduces the pair structure of its parent at one state point and
   fails at another.
3. **Henderson is a statement about equilibrium at a given $(n,T)$.** Drifting has no temperature and a
   hyperparameter-controlled density (the batch size), so "the same state point" is not even well defined.

The lesson is not that identifiability fails but that **the natural theorem does not transfer**, and the physics
literature has spent forty years on precisely this class of difficulty. Anyone attacking `THEORY.md` §15.1 should
read the coarse-graining representability literature first.

---

## 8. Force matching and coarse-graining

`THEORY.md` §12 is, from a physics point of view, the most recognisable part of drifting: it is **force matching**,
a technique with a twenty-year history in molecular coarse-graining.

### 8.1 The same integration by parts

The multiscale coarse-graining (MS-CG) variational principle of Izvekov–Voth and Noid et al. states that the
coarse-grained force field minimising the mean-squared force residual against atomistic forces **is** the gradient of
the many-body potential of mean force. The proof is an integration by parts against the Boltzmann weight, using

$$\int p(\mathbf{y})\,\nabla_{\mathbf{y}}(\cdot)\,d\mathbf{y} = -\int (\nabla_{\mathbf{y}}p)(\cdot)\,d\mathbf{y}
= -\int p\,(\nabla_{\mathbf{y}}\log p)(\cdot)\,d\mathbf{y}.$$

The Drifting Score Identity (`THEORY.md` eq. 49) is the same move, with the test function taken to be the Gaussian
kernel:

$$\mathbb{E}_p\big[k(\mathbf{y}-\mathbf{x})\big] = \tau^2\,\mathbb{E}_p\big[k\,\nabla_{\mathbf{y}}\log p\big].$$

The reason the Gaussian is required is also the same in both settings: the identity needs
$k(\mathbf{y}-\mathbf{x}) = -\tau^2\nabla_{\mathbf{y}}k$, which is a property of the Gaussian and of nothing else.
Notebook 9 §1 verifies the identity converges at $N^{-1/2}$ for the Gaussian and sits at a 40–60% residual for
Laplace.

**Why forces are a better signal than samples.** In coarse-graining this is folklore, and the reasons transfer
exactly:

* A force label is a *local* measurement of $\nabla\log p$ at a single configuration. A sample is one draw from $p$.
  Per configuration, the force carries $d$ numbers of gradient information where the sample carries one bit of
  "here".
* Forces do not require the sampler to have equilibrated *globally*. A configuration from an unconverged trajectory
  still has correct forces; it just has the wrong probability of having been visited. This is why force matching is
  robust to sampling problems that destroy histogram-based methods.
* The catch, also familiar: forces tell you the shape of $\log p$ *locally* and say nothing about the relative depth
  of two basins you never crossed between. Force guidance cannot fix a missing metastable state.

### 8.2 Temperature reweighting is exactly parallel tempering's exponent

The FK prediction $\gamma^\star = 1/k_BT_{\text{target}} - 1/k_BT_{\text{data}}$ (`THEORY.md` eq. 56) is not new
physics — it is the reweighting exponent that appears everywhere in enhanced sampling:

$$\frac{p_{T_2}(\mathbf{x})}{p_{T_1}(\mathbf{x})} \propto \exp\!\Big[-\mathcal{E}(\mathbf{x})\Big(\tfrac{1}{k_BT_2}-\tfrac{1}{k_BT_1}\Big)\Big].$$

This is the acceptance ratio for a replica exchange between two temperatures, the weight in simulated tempering, and
the reweighting factor in WHAM. Drifting's contribution is to notice that inserting this weight into the *kernel*
transports the whole generative model to the new temperature — because the softmax cancels the intractable $\log Z$,
which is the one thing that makes reweighting hard in general.

The cost is also the familiar one: **effective sample size collapses as the temperature gap grows**. Notebook 9 §4
measures ESS falling from 60000 to 30000 for a 2.5× temperature ratio on a 2-D problem; in a system with $10^4$
degrees of freedom the same reweighting would leave you with one effective sample. This is the central difficulty of
all reweighting methods and drifting does not escape it. If you need a large temperature change, you need
FI's direction-based correction or a staged sequence of small tilts — the generative analogue of a temperature
ladder.

### 8.3 Relation to Boltzmann generators

Boltzmann generators (Noé, Olsson, Köhler, Wu) train a normalising flow to sample $e^{-\mathcal{E}/k_BT}$ and then
*reweight* the samples to remove the residual bias, using exact likelihoods. Drifting cannot do that — it has no
tractable likelihood (§10) — so it takes the other route: bake the correction into the training field, and accept
that the equilibrium is approximate.

The trade is legible. Boltzmann generators give asymptotically exact expectations at the price of an invertible
architecture and an ESS that can be catastrophic. Drifting gives a 1-NFE sampler of unrestricted architecture with a
controlled, unremovable bias. For screening or generation, drifting's trade is the better one; for computing a free
energy difference to chemical accuracy, it is not.

---

## 9. Symmetry, gauge fixing, and equivariance

### 9.1 The target measure is invariant; the field must be too

For molecular data, $p_{\text{data}}$ is invariant under the Euclidean group: $p(R\mathbf{X}+t) = p(\mathbf{X})$.
A field that fails to respect this symmetry is not merely inelegant; it makes the model's task harder in a specific,
measurable way, because the model must then learn to undo an artefact the field introduced.

Notebook 8 §1 measures the artefact: a structure and its rotated copy are at Euclidean distance 18.8, while a
genuinely perturbed structure is at 1.9. The kernel confidently reports the identical structure as the more
dissimilar one. Since $\tau$ scales all distances equally, no tuning fixes an inverted ordering.

### 9.2 Kabsch alignment is gauge fixing

The physicist's reading of Drift-React's construction (`THEORY.md` §11): the rigid-body pose is a **gauge degree of
freedom**. Physical content lives in the quotient $\mathbb{R}^{3N}/SE(3)$; a coordinate representation carries six
unphysical parameters.

The recipe is then the standard gauge-theory recipe:

1. **Fix a gauge** — Kabsch-align each structure onto the reference, choosing a canonical representative of each
   orbit.
2. **Compute in the fixed gauge** — form the affinities there. They are functions of aligned distances, hence gauge
   invariant: *scalars*.
3. **Transform back covariantly** — rotate the resulting displacement by $R_i^{\top}$ so it lives in the frame the
   generator emitted. Displacements are *vectors*, so they must transform, and the answer is gauge independent.

The rule "scalars are invariant, vectors are covariant" is the whole content of the construction, and it is why the
equivariance residual in Notebook 8 §3 is floating-point noise ($6\times10^{-15}$ in float64) rather than
approximately small.

The **reflection guard** is the discrete part of the same story. $O(3)$ has two components; chemistry lives in the
identity component $SO(3)$ because molecules are chiral. Allowing $\det R = -1$ silently identifies enantiomers.
Notebook 8 §2 shows the failure: without the guard, an L-form aligns onto a D-form with RMSD $10^{-16}$.

### 9.3 Internal coordinates: working in the quotient

The alternative is to never introduce the gauge: use coordinates that are invariant by construction, e.g. the vector
of pairwise interatomic distances $\phi_k = \|\mathbf{x}_{i_k}-\mathbf{x}_{j_k}\|$.

Physically this is choosing **internal coordinates** — the natural variables of a molecule, the ones a force field is
written in. The costs are the classic ones:

* **The map is not onto.** Not every vector of "distances" corresponds to a real structure; the realisable set is a
  thin, curved manifold cut out by rank and triangle conditions. Notebook 9 §6 measures the resulting damage: even an
  ordinary kernel-weighted *average* of valid distance vectors has a rank-3 embedding residual of 0.024, and
  force interpolation triples it.
* **Chirality is invisible.** Distances are invariant under reflection too, so $\phi$ cannot distinguish enantiomers.
  This is a **discrete symmetry the representation cannot see** — you quotiented by $O(3)$ when you meant $SO(3)$ —
  and it must be restored with a signed quantity (a dihedral, or the sign of a triple product).
* **The Jacobian is sparse but nontrivial**: $\partial\phi_k/\partial\mathbf{x}_a = \pm\hat{\mathbf{r}}_{i_kj_k}$,
  six nonzeros per row, verified exactly in Notebook 8 §5. Cheap, but it is a change of variables and the field's
  geometry changes with it.

### 9.4 Zero modes

After alignment, the six rigid-body directions are **zero modes** of the field: displacing a structure along a global
translation or rotation changes nothing, because alignment removes it. This is desirable — it is exactly the
statement that the field does not waste effort on gauge — and it is also the reason a naive spectral analysis of the
drifting dynamics on molecules will find six vanishing eigenvalues that must be projected out before anything else
makes sense. In vibrational analysis the same six modes are removed before diagonalising the Hessian, for the same
reason.

---

## 10. Non-conservative fields and the impossibility of an energy

### 10.1 Circulation

A vector field is conservative iff its line integral around every closed loop vanishes. For the Laplace drifting
field this fails, and the failure is measurable: Notebook 0a §6 computes the circulation
$\oint\mathbf{V}\cdot d\boldsymbol\ell$ on a closed loop and finds it nonzero, while the same computation for the
Gaussian field gives zero to numerical precision.

Physically: you can extract net work by carrying a test sample around a loop. There is no potential energy function,
because "energy at a point" would have to be path-independent.

### 10.2 Helmholtz decomposition

Any sufficiently regular field splits as

$$\mathbf{V} = -\nabla\Phi + \nabla\times\mathbf{A},$$

a curl-free (gradient) part and a divergence-free (solenoidal) part. For the Gaussian kernel the second term
vanishes; for Laplace it does not. The solenoidal part performs no net descent of any functional — it circulates.

This is the physical content of the extra residual in the finite-particle analysis (`THEORY.md` §7.4). A pure
gradient flow converges monotonically because the functional decreases; a field with a solenoidal component can
circulate indefinitely without decreasing anything, and convergence has to come from the gradient part dominating.
The analysis's "scale-mismatch residual" is what the circulation costs.

### 10.3 Consequences you will actually meet

* **No likelihood, no free energy, no EBM.** You cannot read a density off a drifting model, and there is no
  $\Phi$ to plug into $e^{-\Phi/k_BT}$. Any pipeline that wants to *score* samples, not just generate them, needs a
  separate model.
* **Limit cycles are possible in principle.** A non-conservative field can have periodic orbits rather than fixed
  points. In practice drifting converges — the gradient part dominates in every reported experiment — but "the loss
  is oscillating" is a symptom worth taking more seriously here than in diffusion, where the objective is a genuine
  regression at every step.
* **The Gaussian kernel buys the structure back.** If you need the gradient-flow guarantees — for a convergence
  proof, or for the score identity of §8.1 — use the Gaussian and pay in dynamic range. Drifting's default is a
  deliberate choice to give up structure for performance in high dimensions.

---

## 11. Temperature as resolution: a renormalisation reading

### 11.1 $\tau$ sets the coarse-graining scale

From §3.1, the Gaussian field is $\tau^2\nabla\log(\hat p_\tau/\hat q_\tau)$, where both densities are smeared on
scale $\tau$. So the field at temperature $\tau$ compares the two distributions **as seen at resolution $\tau$**.
Structure finer than $\tau$ is invisible; structure much coarser than $\tau$ is seen but weakly, because the
normalisation removes the overall magnitude.

This is a **blocking transformation** in the Kadanoff sense: smearing on scale $\tau$ is exactly the coarse-graining
step of a real-space renormalisation group, and $\hat p_\tau$ is the coarse-grained density.

### 11.2 Why summing over $\tau$ is not averaging

The paper sums the (individually normalised) fields over several $\tau$:

$$\mathbf{V}_{\text{total}} = \sum_{\tau\in\mathcal{T}} \frac{\mathbf{V}_\tau}{\lambda_\tau}.$$

The RG reading: each scale contributes its own residual, and the total drives agreement **at every scale
simultaneously**. This is not an average — averaging would let a large error at one scale be cancelled by agreement
at another. Because each term is separately normalised to unit RMS, no scale can dominate; each contributes equally,
and only when *all* residuals vanish does the sum vanish.

This is why the per-$\tau$ diagnostic of Notebook 3 §7 is per-$\tau$: an aggregate would hide exactly the situation
the construction was designed to prevent. The characteristic failure — fine scale flat and tiny from step 0 while
coarse scales fall — is the starved kernel of §2.3, and it is visible only per scale. Notebook 7 §2 shows the
*inverse* case, where the fine-scale residual legitimately *rises* over training as the samples spread out enough for
the fine kernel to resolve them at all.

### 11.3 The finest resolvable scale

There is a smallest useful $\tau$: below it, ESS drops to 1 (§7.1), the field becomes nearest-neighbour copying, and
the estimator's variance explodes. In RG language, you cannot coarse-grain below the lattice spacing — and here the
lattice spacing is set by the *sample density*, i.e. by the batch size.

This is the physical reason drifting needs large batches (`THEORY.md` §15.5). The finest structure the model can be
held to is set by the typical nearest-neighbour distance in a batch, which scales as $N^{-1/d}$ in $d$ effective
dimensions. Doubling resolution in a high-dimensional space costs exponentially many samples. **Batch size is a
resolution parameter, not just a variance-reduction parameter**, and this is a qualitative difference from diffusion,
where the resolution comes from the noise schedule and small batches merely add gradient noise.

---

## 12. Finite-particle effects and the N-body problem

### 12.1 Mean field and its corrections

The population field is a mean-field object: each particle responds to the *density* of the others. The finite-$N$
estimator replaces the density with an empirical measure, and the difference is the usual $O(N^{-1/2})$ fluctuation,
verified in Notebook 0b §5.

In the kinetic-theory language, the mean-field limit is the propagation-of-chaos statement: as $N\to\infty$ particles
become independent and each follows the McKean–Vlasov equation. Corrections at finite $N$ are correlations between
particles — the BBGKY hierarchy's second level — and they do not vanish; they are the fluctuations that EMA is
averaging away (Notebook 3 §6, where EMA is the single largest measured improvement).

That is a satisfying identification: **EMA in drifting is doing what time-averaging does in a molecular dynamics
measurement.** You have a noisy instantaneous estimate of an ensemble quantity and you average over a window long
enough to reduce variance but short enough to track drift. And it fails in the same way: an averaging window longer
than the equilibration time averages over the *transient*, which is why a decay of 0.9995 in a 2500-step run is
actively harmful (measured in Notebook 3 §6) and why the decay must be warmed up.

### 12.2 Self-exclusion is self-energy subtraction

Notebook 2 §5 masks the diagonal of the negative affinity matrix so a sample does not repel itself. Physically this
is **self-energy subtraction**, the same step that appears in every lattice sum: in an Ewald summation one must
explicitly remove the interaction of a charge with its own periodic image and with itself, because the naive sum
includes a divergent self-term.

The consequence is quantitatively the same, too. Without the mask, every sample carries a constant self-repulsion of
weight $\bar k(\mathbf{x},\mathbf{x})=1$ (the largest possible), which systematically biases the field. Its size
relative to the real signal is $O(1/N)$ — negligible at $N=8192$, fatal at $N=16$. Small-batch drifting *requires*
the mask; large-batch drifting merely benefits.

### 12.3 The ensemble is the model

The deepest finite-$N$ point: drifting only ever knows the model distribution through $N$ samples. The repulsion term
is computed on a batch, so the "distribution" being pushed around is an $N$-particle ensemble — statistical mechanics
with $N \sim 10^2$–$10^4$ particles, which is a small system by any physical standard.

This explains several otherwise-puzzling facts:

* Why batch structure is a *modelling* decision rather than an engineering one (`THEORY.md` §5.4): the batch defines
  the ensemble the physics happens in.
* Why per-class batches matter for conditional generation: samples of class $a$ must repel other samples of class
  $a$, not of class $b$, or you are simulating the wrong system.
* Why the single-positive regime (Notebook 8 §4) is so different: with one fixed charge and $K$ mobile ones, the
  equilibrium is "all mobile charges at the fixed one" — there is nothing to screen and no spatial structure to
  reproduce. The measured collapse of a 32-particle swarm from spread 0.86 to 0.004 is exactly that.

---

## 13. Where the analogy misleads

Physical intuition is a tool, not an authority. Places where it will actively mislead:

1. **There is no conserved energy.** Do not expect relaxation to a Boltzmann state, do not look for a Lyapunov
   function in the Laplace case, and do not interpret the loss as an energy. The loss is a step size (§6.2).

2. **The "particles" are samples, and their number is a hyperparameter.** In physics, $N$ is given by the system. In
   drifting, changing $N$ changes the interaction itself, because the normalisation depends on it. No physical
   system has this property, and it is why results do not transfer between batch sizes as smoothly as one would hope.

3. **The metric is learned.** In feature-space drifting (Notebook 4), the "space" the particles live in is the output
   of a neural network, so the geometry is not merely non-Euclidean but *chosen*. Distances there have no physical
   dimension; conservation, isotropy, and homogeneity all fail. This is the largest gap between the analogy and the
   practice, and it is why the encoder is the method's biggest liability (`THEORY.md` §15.4).

4. **Anti-symmetry can be — and sometimes should be — broken.** No closed physical system violates Newton's third
   law, but force guidance (§4.2) and Pepti-drift's toxicity repulsion (Notebook 10) both do so deliberately, and
   both work. When the goal is steering rather than equilibrium, the physical constraint is not binding.

5. **"Temperature" appears twice with different meanings.** $\tau$ (a length) and $T$ (an actual temperature, in
   force guidance) are unrelated. Conflating them produces the units bug that Notebook 9 §3 documents, where
   normalising the coordinates before evaluating $\mathbf{F}$ makes the energy move the wrong way with $\omega$.

6. **Screening intuition assumes the mobile charges can move freely.** They cannot: they are the image of a neural
   network with finite capacity. A region can stay unscreened because the network cannot represent the map that
   would screen it — a constraint with no counterpart in a plasma, and one that shows up as "the field is nonzero and
   nothing is improving".

---

## 14. Dictionary

| drifting | physics | where |
|---|---|---|
| positives $\mathbf{y}^{+}$ | quenched charges | §2.1 |
| negatives $\mathbf{y}^{-}$ | mobile like charges | §2.1 |
| $\mathbf{V}^{+}$ | attraction to fixed charges | §2.1 |
| $\mathbf{V}^{-}$ | inter-particle repulsion; entropic force | §5.2 |
| $\mathbf{V}=\mathbf{0}$ | mechanical equilibrium; perfect screening | §2.2, §5.3 |
| $\tau$ | screening length / kernel bandwidth / coarse-graining scale | §2.3, §11.1 |
| $\eta$ | mobility (force $\to$ displacement) | §6.1 |
| $\eta\times$ lr | effective mobility | §6.2 |
| kernel ESS | particles per Debye sphere; plasma parameter | §7.1 |
| KDE $\hat p_\tau$ | smeared charge density | §3.1 |
| $-\tau^2\log\hat p_\tau$ | potential of mean force | §3.1 |
| Gaussian kernel | conservative field, exists a potential | §3.1 |
| Laplace kernel | non-conservative field, no potential | §3.2, §10 |
| softmax normalisation | density-dependent many-body potential | §2.3, §7.3 |
| anti-symmetry $\mathbf{V}_{p,q}=-\mathbf{V}_{q,p}$ | Newton's third law, for distributions | §4.2 |
| $\bar k$ normalisation | nonreciprocal pair interaction | §4.1 |
| identifiability ($\mathbf{V}=0 \Rightarrow q=p$?) | Henderson's theorem and its failure for density-dependent potentials | §7.3 |
| Drifting Score Identity | the force-matching integration by parts | §8.1 |
| force interpolation (FI) | force matching / MS-CG | §8.1 |
| energy tilt $\gamma^\star$ | replica-exchange / WHAM reweighting exponent | §8.2 |
| ESS loss under tilting | reweighting catastrophe | §8.2 |
| Kabsch alignment | gauge fixing | §9.2 |
| $\det R = +1$ guard | $SO(3)$ vs $O(3)$; chirality | §9.2 |
| pairwise-distance features | internal coordinates; the quotient space | §9.3 |
| rigid-body directions | zero modes | §9.4 |
| multi-$\tau$ sum | multi-scale / RG blocking | §11.1 |
| batch size | particle number; sets the resolution floor | §11.3, §12.3 |
| EMA | time-averaging a noisy ensemble measurement | §12.1 |
| self-exclusion mask | self-energy subtraction (Ewald) | §12.2 |
| moving prior | a time-dependent Hamiltonian; no fixed point to relax to | §13.2 of `THEORY.md` |

---

## Further reading

Pointers, in the order the sections use them.

* **Plasma screening and the Debye length** — any plasma-physics text; Chandrasekhar's *Principles of Stellar
  Dynamics* for the gravitational analogue with the opposite sign.
* **Mean shift as a density gradient** — Fukunaga & Hostetler (1975); Comaniciu & Meer (2002).
* **Nonreciprocal interactions and active matter** — Fruchart, Hanai, Littlewood & Vitelli, *Non-reciprocal phase
  transitions*, Nature (2021).
* **Wasserstein gradient flows and JKO** — Jordan, Kinderlehrer & Otto (1998); Ambrosio, Gigli & Savaré, *Gradient
  Flows*.
* **McKean–Vlasov and propagation of chaos** — Sznitman's Saint-Flour lectures.
* **Liquid-state theory, $g(r)$, Ornstein–Zernike** — Hansen & McDonald, *Theory of Simple Liquids*.
* **Henderson's theorem** — Henderson, *Phys. Lett. A* 49 (1974); for the representability failure, Louis,
  *J. Phys.: Condens. Matter* 14 (2002).
* **Force matching / MS-CG** — Izvekov & Voth (2005); Noid et al., *J. Chem. Phys.* 128, 244114 (2008).
* **Reweighting and enhanced sampling** — Ferrenberg & Swendsen (1988) for WHAM; Sugita & Okamoto (1999) for replica
  exchange.
* **Boltzmann generators** — Noé, Olsson, Köhler & Wu, *Science* 365 (2019).
* **Ewald summation and self-energy** — Frenkel & Smit, *Understanding Molecular Simulation*, ch. 12.
* **The physics of diffusion models** — the companion `PHYSICS.md` in the `learning-diffusion` repository, for the
  stochastic-process half of the picture that this document deliberately does not repeat.
