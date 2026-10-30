1. Generative Modeling via Drifting. arXiv:2602.04770
1a. https://lambertae.github.io/projects/drifting/
2. PEPTI-DRIFT: TOXICITY-REPULSIVE DRIFTING FOR ANTIGEN-CONDITIONED DISCRETE PEPTIDE GENERATION. arXiv:2606.27824
3. Drift-React: One-step Generation of Reaction Pathways via SE(3) Drifting Fields. arXiv:2605.22990
3a. https://github.com/schwallergroup/drift-react

---

## Also covered in this repository

The tutorial additionally works through two follow-up papers that `THEORY.md` cites and that the notebooks
reproduce in miniature:

4. Drifting to Boltzmann — score and force guidance toward the Boltzmann distribution (`THEORY.md` §12,
   Notebook 9). arXiv:2603.05527
5. Drift-AR — drifting from a learned, model-dependent prior, and the *moving prior* problem (`THEORY.md` §13.2,
   Notebook 7). arXiv:2603.28049

Notebook 12 and `THEORY.md` §16 draw on the looped-reasoning literature as summarised in the sibling repository
`learning-diffusion` (its `THEORY.md` §12). I used that summary, not the papers:

6. Thinking with Looped Flows. arXiv:2609.11801 (as cited in `learning-diffusion`)
7. Flow Reasoning Models (self-conditioning, Fixed-Point Forcing). arXiv:2606.29150 (as cited in `learning-diffusion`)
8. One Step Forward and K Steps Back (Denoising Recursion Models). arXiv:2604.18839 (as cited in `learning-diffusion`)
9. Tiny Recursive Models (Jolicoeur-Martineau, 2025) and Hierarchical Reasoning Models (Wang et al., 2025), the looped
   baselines those papers build on; self-conditioning is due to Chen et al. (2022). Citations as given in `learning-diffusion`.
10. Tweedie's formula (Efron, 2011, *Tweedie's formula and selection bias*), used in Notebook 0a and `THEORY.md` §7.1.

arXiv identifiers for items 1–8 are as recorded in `THEORY.md` or `learning-diffusion`; verify them against the published
versions before citing.

## Repository documents

* `THEORY.md` — the mathematics
* `PHYSICS.md` — the same objects as an interacting-particle system
* `PROTENIX_DRIFTING.md` — applying drifting to Protenix v2
* `README.md` — the tour of the notebooks
