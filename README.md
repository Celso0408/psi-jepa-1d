# Psi-Jepa-1d

Topology-preserving joint-embedding predictive learning for real, one-dimensional
self-adjoint Schrödinger operators.

This repository is centered on one self-contained Google Colab/Jupyter notebook:

**[`Schrodinger_True_JEPA_1D_Colab_v6_Topology.ipynb`](Schrodinger_True_JEPA_1D_Colab_v6_Topology.ipynb)**

The notebook predicts the lowest eleven ordered energies and signed real wavefunctions
from a complete operator specification. Its default direct decoder uses a strictly
positive amplitude and a strictly monotone phase, enforcing the Sturm node count for
state `n` by construction. An older spectral sine-residual decoder remains available as
a baseline, and a safeguarded Rayleigh–Ritz/refinement route is reported separately as a
hybrid result.

> **Research status:** this is an auditable scientific machine-learning research
> notebook, not an exact or universal quantum solver. `QUICK` is a smoke experiment and
> must not be presented as scientific evidence. No architecture-superiority or OOD
> generalization claim is made without completed matched experiments.

## Scientific problem

The notebook models

$$
H[V]\psi_n(x)
=\left[-\kappa\frac{d^2}{dx^2}+V(x)\right]\psi_n(x)
=E_n\psi_n(x),
\qquad n=0,\ldots,10,
$$

with default kinetic coefficient \(\kappa=\hbar^2/(2m)=1/2\). Wavefunctions are
quadrature-normalized, globally sign-aligned for comparison, and densities are always
defined by the identity

$$
\rho_n(x)=\psi_n(x)^2.
$$

There is no independent density head.

## Topology-preserving decoder

Using the dimensionless coordinate \(t=(x-a)/(b-a)\), the default
`TopologyPhaseDecoder` predicts nodewise log-amplitude and phase-density fields from
state-conditioned spatial latents. It constructs

$$
q_n(t)>0,\qquad
C_n(t)=\int_0^t q_n(s)\,ds,\qquad
C_n(0)=0,\quad C_n(1)=1,
$$

and

$$
\theta_n(t)=(n+1)\pi C_n(t),
\qquad
\phi_n(t)=A_n(t)\sin\theta_n(t),
\qquad A_n(t)>0.
$$

Because the phase is strictly increasing from `0` to `(n+1)π`, the continuous direct
wavefunction has exactly `n` interior nodes. Boundary zeros and physical normalization
are imposed structurally. The implementation uses the actual nonuniform quadrature,
excludes padded intervals, and includes a phase-resolution gate because a continuous
node guarantee does not by itself ensure that every lobe is adequately sampled on a
finite grid.

This guarantee is restricted to scalar, real, one-dimensional self-adjoint
Sturm–Liouville/Schrödinger problems with the supported separated
Dirichlet/Friedrichs boundary conditions. It is not claimed unchanged for periodic
boundaries, coupled-channel or matrix-valued potentials, non-Hermitian operators,
complex states, higher dimensions, continuum states, or unsupported internal
singularities.

## Architecture

```text
operator/grid specification only
    -> potential graph encoder (GNN–KAN by default)
    -> state-conditioned JEPA predictor
       -> node latents z_nodes -> topology phase decoder -> psi_direct -> rho_direct
       -> pooled state tokens  -> operator-scaled energy head -> ordered energies

exact psi (training only)
    -> online solution encoder
    -> EMA target encoder (stop-gradient JEPA target)

psi_direct
    -> optional weighted orthogonalization / Rayleigh–Ritz / residual refinement
    -> separately reported hybrid output or topology-safe fallback
```

The energy head predicts an operator-scaled ground-energy correction and positive
adjacent log-gap ratios around a target-free sine–Galerkin reference. It is trained with
exact-energy supervision and consistency with the direct weak Rayleigh quotient.

## Target separation and inference contract

The potential encoder, predictor, direct decoder, and learned energy head receive only
operator information.

| Operator-side inputs | Training targets kept outside the direct route |
|---|---|
| coordinates and domain endpoints | exact wavefunctions |
| potential samples | exact energies and densities |
| quadrature and grid spacing | exact node positions |
| kinetic coefficient | solution-derived state masks |
| geometry and boundary metadata | online/EMA target latents |
| node, interval, padding and singularity masks | generator-only hidden parameters |

Exact solutions are used only by the online/EMA solution branch and supervised losses.
The notebook includes tests showing that target tensors are absent from inference and
that altering target-only node or solution tensors cannot change `forward_operator`.

## Direct and hybrid outputs

| Route | Meaning | Topology status |
|---|---|---|
| `direct_topology` | Pure learned amplitude–phase surrogate | Exactly `n` continuous interior nodes by construction |
| `initial_ritz` | Weighted orthogonalized and Rayleigh–Ritz rotated trial space | Audited separately; topology can change |
| `hybrid` | Optional recurrent residual refinement of the Ritz result | Accepted only after residual, variational and topology safeguards |

The hybrid route never silently replaces the direct topology result. If a Ritz rotation
or refinement proposal loses the required node count, collapses a lobe, becomes
non-finite, or fails the existing physics safeguards, the notebook retains the direct
prediction and records a fallback flag.

## What the notebook contains

- float64 mapped P1 finite-element reference solver with Gauss–Legendre potential
  integration;
- analytical and numerical convergence checks;
- diverse analytical, random, multiwell, barrier, defect, radial and near-continuum
  potential families;
- group-safe train/validation/test splitting, fingerprints and near-duplicate audits;
- padded graph batching for uniform and nonuniform grids;
- direct, GNN–MLP and GNN–KAN backbones with Chebyshev, Legendre and Hermite bases;
- online and EMA solution encoders for true cross-domain JEPA training;
- topology phase and spectral sine-residual decoder ablations;
- operator-scaled ordered energy prediction and Rayleigh consistency;
- waveform, density, energy, gap, H1, residual, orthogonality, subspace, topology,
  reflection and latent-collapse losses;
- per-state physics authorization with hysteresis and gradual ramping;
- topology-aware Rayleigh–Ritz and recurrent-refinement safeguards;
- target-free inference, serialization and checkpoint compatibility tests;
- ID, parameter-OOD and functional-form-OOD evaluation with category-separated metrics;
- direct-versus-hybrid dashboards, topology figures, CSV exports, run card and model
  card.

## Running the notebook

### Google Colab

1. Open or upload
   [`Schrodinger_True_JEPA_1D_Colab_v6_Topology.ipynb`](Schrodinger_True_JEPA_1D_Colab_v6_Topology.ipynb).
2. Select a GPU runtime for modes larger than `QUICK`.
3. Choose the run mode before executing the configuration cell.
4. Restart the runtime after changing modes, then use **Runtime → Run all**.

The notebook currently defaults to `DEVELOPMENT`. For a first smoke execution, add a
small cell before the notebook configuration is evaluated:

```python
import os
os.environ["PSI_JEPA_RUN_MODE"] = "QUICK"
os.environ["PSI_JEPA_SEED"] = "7"
```

Do not switch mode halfway through a run or execute only a suffix of the notebook. The
dataset, configuration, checkpoints and run card are intentionally hash-bound.

### Local Jupyter

Install Python with PyTorch, NumPy, SciPy, pandas, Matplotlib and Jupyter. PyTorch
Geometric is not required. Start the kernel with the mode and seed already defined:

```bash
PSI_JEPA_RUN_MODE=QUICK \
PSI_JEPA_SEED=7 \
PYTHONHASHSEED=7 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
jupyter lab
```

Open the notebook and run all cells in order from a fresh kernel.

## Run modes

All modes use the same solver, topology decoder, losses, inference routes, mandatory
tests and evaluation code. Larger modes increase dataset size, network width, grid
resolution and training budget.

| Mode | Intended use | Hamiltonians per family | Reference/model nodes | Epoch budget: AE / JEPA / supervised / physics / fine-tune |
|---|---|---:|---:|---:|
| `QUICK` | End-to-end smoke test | 3 | 513 / 129 | 8 / 1 / 3 / 1 / 1 |
| `DEVELOPMENT` | Iterative GPU experimentation | 32 | 513 / 129 | 24 / 6 / 18 / 12 / 8 |
| `STANDARD` | Substantive training run | 128 | 1025 / 257 | 20 / 6 / 30 / 15 / 10 |
| `FULL` | Highest configured single-run budget | 192 | 2049 / 513 | 28 / 10 / 40 / 20 / 12 |

`QUICK` verifies wiring and execution only. Scientific conclusions require converged
larger runs, untouched ID testing, category-separated OOD evaluation, multiple seeds
and matched ablations.

## Embedded DEVELOPMENT execution

The notebook currently committed in this workspace contains a complete `DEVELOPMENT`,
seed-7 execution (`true_jepa_development_cfaf3ccf_1e1bcfa6`). Its recorded status is:

```text
DEVELOPMENT_RUN_COMPLETED_NOT_SCIENTIFICALLY_ACCEPTED
```

| Measurement | Embedded result | Interpretation |
|---|---:|---|
| Mandatory tests | 70 / 70 | Pass |
| Topology acceptance requirements | 17 / 17 | Pass |
| Solution-AE reconstruction fidelity | 0.999999 | Pass; training-only target branch |
| Direct continuous Sturm topology | 1.000000 | Pass by construction |
| Direct persistent sampled topology | 0.910954 | Diagnostic; resolution dependent |
| Untouched-ID mean fidelity | 0.757175 | Fails the notebook's scientific target |
| Untouched-ID mean relative H1 | 0.452857 | Fails the scientific target |
| Untouched-ID mean independent residual | 0.310183 | Fails the scientific target |
| Untouched-ID mean normalized energy error | 2.67085 | Fails the scientific target |
| Untouched-ID maximum post-orthogonalization Gram error | 0.0495572 | Fails the scientific target |
| Hybrid direct-fallback fraction | 0.979508 | Diagnostic; hybrid is rarely accepted |
| Unseen analytical functional-form mean fidelity | 0.43833 | Diagnostic only; not OOD evidence |

This execution demonstrates that the numerical pipeline, target separation, exact
continuous topology, checkpointing, evaluation and exports run end to end. It does not
establish scientifically satisfactory waveform, energy, residual, orthogonality,
sampled-node-position or unseen-functional-form performance. It is one seed and does
not include completed matched baseline/ablation experiments.

## Outputs

Each execution writes to

```text
psi_true_jepa_topology_v6_outputs/<experiment_name>/
```

Important artifacts include:

- `config.json`, `run_card.json` and `model_card.json`;
- `solution_autoencoder.pt`, `best.pt` and `last.pt`;
- `test_results.csv` and the topology mandatory-test matrix;
- training history and gradient diagnostics;
- per-state direct, initial-Ritz and hybrid metrics;
- topology, node-position, lobe-mass and refinement audits;
- category-, family-, state- and geometry-level summaries;
- ID/OOD and covariance results;
- the Colab overview and topology-focused figures.

The notebook restores a validation-selected checkpoint before final evaluation. A low
training loss alone is never treated as scientific success.

`CFG.clean_output=True` by default. Re-running the same source, configuration and seed
reuses the same experiment name and removes that experiment directory before writing
new artifacts. Copy any run you want to preserve before starting an identical run.

## Interpreting results

- Autoencoder fidelity measures reconstruction when the exact wavefunction is supplied
  to the training-only solution branch; it is not target-free surrogate fidelity.
- Direct topology compliance establishes the Sturm node count, not waveform, energy,
  residual or OOD accuracy.
- The physics ramp printed in training logs is the maximum statewise ramp; a value of
  `1.0` does not imply that every state is fully physics-authorized.
- Direct, initial-Ritz and hybrid metrics must be compared as different computational
  routes.
- OOD categories must not be pooled into a single headline number.
- No claim of superiority is justified until matched seeds, splits, budgets, parameter
  counts and ablations have actually been executed.

## Limitations

- Only states `n=0,...,10` are modeled.
- Results depend on finite-grid resolution, reference-solver accuracy and the sampled
  training distribution.
- Singular endpoints, radial mappings, truncated domains and near-continuum states are
  numerically difficult.
- Exact phase topology does not guarantee accurate node positions or lobe probability
  masses on an under-resolved grid.
- Orthogonalization and Rayleigh–Ritz rotation can alter individual-state topology;
  safeguards therefore report fallback rather than hiding failure.
- The notebook does not establish universal generalization to unseen operators.
- `STANDARD` and `FULL` can exceed the practical duration of a single Colab session.

## Suggested repository description

> Topology-preserving JEPA surrogate for real 1D self-adjoint Schrödinger operators,
> featuring exact Sturm node counts, target-free inference, GNN–KAN backbones, and
> optional Rayleigh–Ritz refinement.
