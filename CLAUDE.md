# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this repository is

A self-contained tutorial on **drifting** — one-step generative modelling via training-time distribution transport —
plus a design document applying it to Protenix. It is a *teaching* repository: the notebooks are the product, and
their job is to make claims that a reader can check. Prose that asserts without measuring is a defect here.

There is no library, no package to install, no test suite. `pyproject.toml` exists only to pin the notebook
dependencies.

## Setup

```bash
uv sync
source .venv/bin/activate
jupyter lab
```

Python $\ge$ 3.12, PyTorch $\ge$ 2.9. Notebooks detect CUDA and fall back to CPU. MNIST downloads to `./data`
(gitignored).

## Layout

```
THEORY.md               the mathematics; the reference the notebooks cite by equation number
PHYSICS.md              the same objects read as an interacting-particle system
PROTENIX_DRIFTING.md    design doc: drifting inside Protenix v2. No code, by request.
README.md               the tour
REFERENCES.md           source papers

0a/0b/0c *.ipynb        prerequisites: kernels & fields, pushforwards & distances, fixed points & stop-gradients
1-5 *.ipynb             core method: basics, field anatomy, practice, feature space, conditioning + CFG
6 *.ipynb               comparison with diffusion / flow / GAN / MMD
7 *.ipynb               latent drifting and learned priors
8 *.ipynb               SE(3) equivariant drifting
9 *.ipynb               score and force guidance toward Boltzmann
10 *.ipynb              discrete drifting for peptides
```

Filenames contain spaces and are linked from within notebooks as `[title](<N. Title.ipynb>)`. **Do not rename them
without updating the cross-references**, which appear at the end of most notebooks and inside `README.md`.

## The house style

Every notebook follows the same shape, and new material should match it.

* **A claim, then the measurement that supports it.** Numbers in prose must come from a cell in the same notebook.
  If a number is quoted from a paper, say so.
* **Prose carries the argument; code is the evidence.** Markdown cells are long and explanatory; code cells are
  short, dense, and free of decoration.
* **Ablations over assertions.** "$X$ matters" is established by running with and without $X$ and printing both.
* **Failure modes are content.** A negative result — the single-positive collapse in notebook 8, the mode collapse
  in notebook 10, the moving prior in notebook 7 — is more valuable than another success, and gets the same care.
* **Cross-reference precisely.** Notebooks cite `THEORY.md` by section and equation number, and cite each other by
  notebook and section. Keep those accurate when editing either side.
* **Notebooks are stored without outputs.** `execution_count: null`, empty `outputs`. Run them to see results;
  do not commit executed copies.

## Shared code conventions

The same handful of functions recurs across notebooks, deliberately re-defined in each rather than imported, so that
every notebook stands alone. Keep the signatures consistent:

```python
compute_V(x, y_pos, y_neg=None, tau=0.2)   # the field; y_neg=None means self-negatives with a masked diagonal
feature_scale(f, fp)                        # S_j from THEORY.md eq. 21
normalised_drift(V)                         # V / RMS(V), THEORY.md eq. 23-25
```

Other recurring conventions:

* `device = "cuda" if torch.cuda.is_available() else "cpu"` at the top of every notebook.
* `torch.manual_seed(0)`, `plt.rcParams["figure.dpi"] = 110`.
* The joint normalisation is always $A = \sqrt{\sigma_{\text{row}}(L)\,\sigma_{\text{col}}(L)}$ — row *and* column.
* Self-exclusion is a $-10^{6}$ additive mask on the diagonal, never a boolean index.
* Multi-temperature sums use per-$\tau$ drift normalisation *before* summing, never after.
* Geometry checks (notebooks 8 and 9) run in `float64`; training runs in `float32`.

## Things that are easy to get wrong

Collected because each of these has already caused a wrong result in this repository.

1. **The repulsion sign.** The drift is $\mathbf{V}^{+} - \mathbf{V}^{-}$ where $\mathbf{V}^{-}$ is the mean-shift
   vector *toward* the negatives. Written in the "push away" form of `THEORY.md` eq. 59 the sign flips, and getting
   it backwards attracts samples to exactly what they should avoid — which trains stably and looks plausible.
2. **Feature normalisation and physical units.** Notebook 9's force guidance evaluates $\mathbf{F}$ and
   $\mathcal{E}$ at coordinates. Applying notebook 3's feature-scale normalisation first evaluates them at the wrong
   physical point; the symptom is that guidance moves the ensemble the wrong way, not a crash.
3. **The training loss is not a convergence diagnostic** once the drift is normalised — it is pinned near
   $\eta^2$. Log the un-normalised $\|\mathbf{V}\|^2$ per temperature.
4. **Kabsch without the $\det=+1$ guard** aligns enantiomers onto each other and reports RMSD $10^{-16}$.
5. **A single positive makes the kernel inert.** $\bar k \equiv 1$, so $\mathbf{V}^{+} = \mathbf{y}^{\star}-\mathbf{x}$
   regardless of $\tau$. If a change makes the field $\tau$-independent, that is the bug.
6. **EMA decay must match the run length.** A decay of 0.9995 has a 2000-step horizon and actively hurts a
   2500-step run. Warm the decay up: $d_s = \min(d, (1+s)/(10+s))$.
7. **`torch.histogramdd` needs CPU tensors.** The `tv2d` helpers call `.cpu()` for this reason.

## Editing

* **Notebooks are generated from builder scripts during authoring**, then committed as `.ipynb`. When editing an
  existing notebook, edit the `.ipynb` directly — the builders are not kept in the repository.
* Prefer editing prose over adding cells. These notebooks are already long; a new section should replace something
  or earn its place.
* If you change a number in a code cell, **re-run the notebook and update every number quoted in the prose around
  it**. Stale quoted numbers are the most common defect in a repository like this one.
* `THEORY.md` equation numbers are referenced from the notebooks. Adding an equation in the middle renumbers
  everything after it; add at the end of a section or accept the renumbering work.

## Verifying a change

There is no test suite, so verification is: run the affected notebook end to end, and check that the printed numbers
still match the prose. A convenient way to run one headlessly is to execute its code cells in order with a matplotlib
`Agg` backend, which surfaces exceptions without needing a display.

Notebooks 4 and 7 train small networks on MNIST and take a few minutes on a GPU; the rest are faster.
