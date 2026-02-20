# Methodology

## Fairness Framework

This audit follows the framework established in:
- Hardt, Price, Srebro (2016) — "Equality of Opportunity in Supervised Learning"
- Barocas, Hardt, Narayanan (2019) — "Fairness and Machine Learning"

### Metric Definitions

**Demographic Parity**: A classifier satisfies demographic parity if the probability of
a positive prediction is the same across all groups.

**Equalized Odds**: A classifier satisfies equalized odds if the true positive rate and
false positive rate are equal across all groups.

**Predictive Parity**: A classifier satisfies predictive parity if the positive predictive
value (precision) is equal across all groups.

**Disparate Impact (4/5 Rule)**: The ratio of favorable outcome rates between the
unprivileged and privileged groups must be at least 0.80 (80%).

## Causal Inference

### Difference-in-Differences (DiD)

Exploits temporal variation in lending policies. The parallel trends assumption requires
that treatment and control groups would have followed similar outcome trajectories in
the absence of treatment.

### Instrumental Variables (2SLS)

Uses state-level unemployment as an instrument for individual income. Validity requires:
1. **Relevance**: Unemployment correlates with income (tested via first-stage F-stat)
2. **Exclusion**: Unemployment affects default only through income (untestable, argued economically)

## Bayesian Approach

Uses Beta-Binomial conjugate models for group-level positive prediction rates.
The posterior of the fairness gap is computed via Monte Carlo sampling.
95% Highest Density Intervals (HDI) provide credible intervals for the gap.
