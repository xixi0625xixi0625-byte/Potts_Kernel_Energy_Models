## potts_kernel_energy_models
Energy-based models for biological sequences: Potts vs Kernel energy, fit with exact likelihood or pseudolikelihood, ML or Bayesian (MAP), with epistasis analysis. 

The model is p(x) = exp(E(x)) / Z, and we compare different choices for the energy E, the objective, and the inference.

## What is compared
**Energy**
*Potts*: 'E(x) = sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)'
*Kernel*: 'E(x) = sum_u K(x, u) alpha_u', with 'K(x, u) = sigma_E^2 * exp(-H(x, u) / c)' (H = Hamming distance)

**Objective**
*Exact likelihood*: Z computed by enumerating all sequences
*Pseudolikelihood*: 'sum_n sum_i log p(x_i|x_-i)'

**Inference**
*ML*: maximize the log likelihood
*BI*: maximize the log likelihood and the log prior (MAP)
  - Potts prior: Gaussian on h and J
  - Kernel prior: alpha ~ N(0, K_uu^-1)

All 8 combinations are fit with L-BFGS-B using analytic gradients.

## Outputs
- Average log p(x) on the data
- Coupling strength 'J_ij' (Potts only)
- Epistasis between position pairs, 'epsilon = (e11 - e01) - (e10 - e00)', averaged over data background.

## Data
60 DNA sequences of length 6 (alphabet  ACGT), defined in the notebook.

## Files
- 'Potts_Kernel_Energy_Models.ipynb': main notebook (models, fitting, results)
- 'tests/': unit tests

## Notes
sigma_E^2 and c in the Kernel are fixed for now and will be learned from the data later.
