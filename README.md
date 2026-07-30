# mojo-lifelines

Survival-analysis fitters implemented in [Mojo](https://www.modular.com/mojo)
and exposed to Python with lifelines-compatible class names, constructor
arguments, `fit` signatures, fitted attributes, and prediction methods.

```python
import pandas as pd
from mojo_lifelines import CoxPHFitter, KaplanMeierFitter

km = KaplanMeierFitter().fit([1, 2, 3, 4], [1, 0, 1, 1])
print(km.survival_function_)

df = pd.DataFrame({
    "time": [1, 2, 3, 4, 5, 6],
    "event": [1, 1, 0, 1, 0, 1],
    "treatment": [0, 1, 0, 1, 0, 1],
    "age": [42, 51, 39, 63, 48, 57],
})
cox = CoxPHFitter().fit(df, duration_col="time", event_col="event")
print(cox.params_)
```

This is a standalone implementation, not bindings to Python lifelines.
Upstream `lifelines` is installed only as a development dependency for parity
tests and benchmarks.

## Coverage

| fitter | covered behavior |
| --- | --- |
| `KaplanMeierFitter` | right-censored observations, ties, integer case weights, custom timelines, Greenwood log-log confidence intervals, event table |
| `NelsonAalenFitter` | right-censored observations, ties, smooth and discrete increments, confidence intervals |
| `BreslowFlemingHarringtonFitter` | survival estimate from the smoothed Nelson–Aalen cumulative hazard |
| `ExponentialFitter` | censored and weighted maximum likelihood, fitted distribution methods and variance |
| `WeibullFitter` | censored and weighted maximum likelihood, fitted distribution methods and observed-information variance |
| `CoxPHFitter` | numeric covariates, Efron ties, Breslow baseline, coefficients and inference, partial-hazard and survival predictions, concordance and log-likelihood scoring |

The covered APIs preserve lifelines' `fit` parameter names and return pandas
`Series` and `DataFrame` objects with the expected labels. The test suite
compares every listed fitter directly with upstream on representative
right-censored data. Separate tests cover tied failures, censoring, integer
weights, custom timelines, confidence intervals, parametric and Cox variance
matrices, Cox baselines, predictions, scores, SIMD remainders, invalid native
inputs, and public `fit` signatures.

Not covered are left or interval censoring, delayed entry, stratified or
time-varying Cox models, formulas, cluster or robust covariance, L1 penalties,
parametric spline baselines, competing-risks fitters, AFT fitters, and plotting.
Unsupported fit options raise `NotImplementedError` instead of silently
changing the statistical model.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` provides the pinned Mojo nightly, Python, NumPy, pandas, SciPy,
pytest, and upstream lifelines. `pixi run build` creates
`dist/libmojo-lifelines.so`. To use a library built elsewhere, set
`MOJO_LIFELINES_LIB` to its absolute path.

## Performance

Measured against upstream lifelines on the same arrays and process. Each
number is the best of three warmed runs from `pixi run bench`, whose Pixi task
holds a machine-wide lock.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz.

| case | mojo-lifelines | lifelines | result |
| --- | ---: | ---: | ---: |
| `KaplanMeierFitter.fit` (500k, tied) | 52.1 ms | 59.5 ms | 1.14x faster |
| `NelsonAalenFitter.fit` (500k, tied) | 65.6 ms | 122.4 ms | 1.87x faster |
| `WeibullFitter.fit` (100k) | 40.4 ms | 1086.6 ms | 26.88x faster |
| `CoxPHFitter.fit` (20k x 8) | 266.0 ms | 2496.0 ms | 9.39x faster |
| Cox `predict_partial_hazard` (300k x 8) | 25.6 ms | 255.7 ms | 9.98x faster |

These are end-to-end estimator calls, including pandas result construction and
sorting where the API requires it. The Cox fit retains NumPy only for its small
dense Newton solve; the full risk-set gradient and Hessian are timed on the
Mojo side. Nelson–Aalen variance is accumulated in the same Mojo pass as its
hazard, avoiding a Python loop over failures.

No GPU path is provided.

## How it works

`src/lifelines.mojo` is one compilation unit with four C exports. The
event-table export sorts nowhere and allocates nothing: Python supplies
duration-sorted contiguous `float64` buffers, and Mojo writes removals,
failures, censoring, at-risk counts, Kaplan–Meier products, Greenwood sums, and
Nelson–Aalen sums and variances in one pass. Native-width SIMD loads reduce
weights and observed-event weights, with scalar tails for remainder elements.
The Weibull kernel solves the profiled likelihood equation for shape and scale
and returns the observed-information covariance.

The Cox export walks event times backwards, maintaining the scalar, vector,
and matrix risk-set moments. For every tied failure set it accumulates the
Efron partial log likelihood, gradient, and Hessian in one call. Python
performs the small Newton solve and constructs the pandas-facing fitted
objects.

All arrays are C-contiguous, row-major `float64`. The ctypes layer passes their
addresses as 64-bit integers; each `@export(... ) abi("C")` wrapper rebuilds an
`UnsafePointer[Float64, AnyOrigin[mut=True]]`. Python owns every input, output,
and scratch allocation, so no ownership or allocator crosses the FFI boundary.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark must be run through Pixi so the machine-wide flock remains in
effect.

## License

MIT
