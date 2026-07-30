from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._lib import addr, f64, lib


def _inputs(durations, event_observed=None, weights=None, entry=None):
    durations = f64(durations)
    if durations.ndim != 1 or durations.size == 0:
        raise ValueError("durations must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError("durations must be finite and non-negative")
    events = np.ones(durations.size, dtype=np.float64) if event_observed is None else f64(event_observed)
    if events.shape != durations.shape or not np.all((events == 0) | (events == 1)):
        raise ValueError("event_observed must contain one boolean or 0/1 value per duration")
    sample_weights = np.ones(durations.size, dtype=np.float64) if weights is None else f64(weights)
    if (
        sample_weights.shape != durations.shape
        or not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights <= 0)
    ):
        raise ValueError("weights must be finite, positive, and match durations")
    if entry is not None and np.any(f64(entry) != 0):
        raise NotImplementedError("delayed entry is not covered by mojo-lifelines")
    return durations, events, sample_weights


@dataclass
class _EventScan:
    event_table: pd.DataFrame
    survival: pd.Series
    greenwood: pd.Series
    na_smooth: pd.Series
    na_discrete: pd.Series
    na_smooth_variance: pd.Series
    na_discrete_variance: pd.Series


def _event_scan(durations, events, weights) -> _EventScan:
    if np.min(durations) > 0:
        durations = np.r_[0.0, durations]
        events = np.r_[0.0, events]
        weights = np.r_[0.0, weights]
    order = np.argsort(durations, kind="stable")
    durations = np.ascontiguousarray(durations[order])
    events = np.ascontiguousarray(events[order])
    weights = np.ascontiguousarray(weights[order])
    size = durations.size
    arrays = [np.empty(size, dtype=np.float64) for _ in range(11)]
    count = lib().ml_event_table(
        addr(durations),
        addr(events),
        addr(weights),
        size,
        *(addr(a) for a in arrays),
    )
    if count < 1 or count > size:
        raise RuntimeError(f"native event scan returned invalid row count {count}")
    (
        times,
        removed,
        observed,
        censored,
        at_risk,
        survival,
        greenwood,
        smooth,
        discrete,
        smooth_variance,
        discrete_variance,
    ) = (a[:count] for a in arrays)
    index = pd.Index(times, name="event_at", copy=False)
    entrance = np.zeros(count)
    entrance[0] = weights.sum()
    table = pd.DataFrame(
        {
            "removed": removed,
            "observed": observed,
            "censored": censored,
            "entrance": entrance,
            "at_risk": at_risk,
        },
        index=index,
        copy=False,
    )
    return _EventScan(
        table,
        pd.Series(survival, index=index, copy=False),
        pd.Series(greenwood, index=index, copy=False),
        pd.Series(smooth, index=index, copy=False),
        pd.Series(discrete, index=index, copy=False),
        pd.Series(smooth_variance, index=index, copy=False),
        pd.Series(discrete_variance, index=index, copy=False),
    )


def _timeline_series(values: pd.Series, timeline) -> pd.Series:
    if timeline is None:
        return values
    requested = np.sort(f64(timeline))
    combined = values.index.union(requested)
    return values.reindex(combined).ffill().reindex(requested).fillna(values.iloc[0])


def _predict_frame(frame: pd.DataFrame, times, interpolate=False):
    scalar = np.isscalar(times)
    query = np.atleast_1d(np.asarray(times, dtype=np.float64))
    base = frame.iloc[:, 0]
    if interpolate:
        union = base.index.union(query)
        result = base.reindex(union).interpolate("index").reindex(query)
    else:
        result = base.reindex(base.index.union(query)).ffill().reindex(query)
    result.index = query
    return float(result.iloc[0]) if scalar else result


class _Univariate:
    alpha: float

    def predict(self, times, interpolate=False):
        return _predict_frame(getattr(self, self._estimation_method), times, interpolate)

    def survival_function_at_times(self, times, label=None):
        values = np.asarray(times, dtype=np.float64)
        result = pd.Series(self._survival(values), index=values)
        result.name = label or self._label
        return result

    def cumulative_hazard_at_times(self, times, label=None):
        values = np.asarray(times, dtype=np.float64)
        result = pd.Series(self._cumulative_hazard(values), index=values)
        result.name = label or self._label
        return result

    def hazard_at_times(self, times, label=None):
        values = np.asarray(times, dtype=np.float64)
        result = pd.Series(self._hazard(values), index=values)
        result.name = label or self._label
        return result


class KaplanMeierFitter(_Univariate):
    def __init__(self, alpha: float = 0.05, label: str = None):
        self.alpha = alpha
        self._label = label

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label=None,
        alpha=None,
        ci_labels=None,
        weights=None,
        fit_options=None,
    ):
        if fit_options:
            raise NotImplementedError("Kaplan-Meier fit_options are not covered")
        self.durations, self.event_observed, self.weights = _inputs(
            durations, event_observed, weights, entry
        )
        self.entry = np.zeros_like(self.durations) if entry is None else f64(entry)
        scan = _event_scan(self.durations, self.event_observed, self.weights)
        self.event_table = scan.event_table
        self.timeline = (
            scan.survival.index.to_numpy()
            if timeline is None
            else np.sort(f64(timeline))
        )
        self._label = label or self._label or "KM_estimate"
        estimate = _timeline_series(scan.survival, timeline)
        variance = _timeline_series(scan.greenwood, timeline)
        self.survival_function_ = estimate.to_frame(self._label)
        self.cumulative_density_ = (1.0 - estimate).to_frame(self._label)
        self._cumulative_sq_ = variance
        level = self.alpha if alpha is None else alpha
        z = norm.ppf(1.0 - level / 2.0)
        s = estimate.to_numpy()
        g = variance.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v = np.log(s)
            lower = np.exp(-np.exp(np.log(-v) - z * np.sqrt(g) / v))
            upper = np.exp(-np.exp(np.log(-v) + z * np.sqrt(g) / v))
        lower = np.nan_to_num(lower, nan=1.0)
        upper = np.nan_to_num(upper, nan=1.0)
        names = ci_labels or [
            f"{self._label}_lower_{1-level:g}",
            f"{self._label}_upper_{1-level:g}",
        ]
        self.confidence_interval_ = pd.DataFrame(
            {names[0]: lower, names[1]: upper}, index=estimate.index
        )
        self.confidence_interval_survival_function_ = self.confidence_interval_
        self.confidence_interval_cumulative_density_ = pd.DataFrame(
            {names[0]: 1.0 - upper, names[1]: 1.0 - lower}, index=estimate.index
        )
        crossed = estimate.index[estimate <= 0.5]
        self.median_survival_time_ = float(crossed[0]) if len(crossed) else np.inf
        self._estimation_method = "survival_function_"
        return self

    def _survival(self, times):
        return np.asarray(_predict_frame(self.survival_function_, times))

    def _cumulative_hazard(self, times):
        raise NotImplementedError("Kaplan-Meier does not estimate cumulative hazard")

    def _hazard(self, times):
        raise NotImplementedError("Kaplan-Meier does not estimate hazard")


class NelsonAalenFitter(_Univariate):
    def __init__(self, alpha=0.05, nelson_aalen_smoothing=True, **kwargs):
        self.alpha = alpha
        self.nelson_aalen_smoothing = nelson_aalen_smoothing
        self._label = kwargs.pop("label", None)

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label=None,
        alpha=None,
        ci_labels=None,
        weights=None,
        fit_options=None,
    ):
        if fit_options:
            raise NotImplementedError("Nelson-Aalen fit_options are not covered")
        self.durations, self.event_observed, self.weights = _inputs(
            durations, event_observed, weights, entry
        )
        self.entry = np.zeros_like(self.durations) if entry is None else f64(entry)
        scan = _event_scan(self.durations, self.event_observed, self.weights)
        self.event_table = scan.event_table
        base = scan.na_smooth if self.nelson_aalen_smoothing else scan.na_discrete
        estimate = _timeline_series(base, timeline)
        base_variance = (
            scan.na_smooth_variance
            if self.nelson_aalen_smoothing
            else scan.na_discrete_variance
        )
        variance = _timeline_series(base_variance, timeline)
        self.timeline = estimate.index.to_numpy()
        self._label = label or self._label or "NA_estimate"
        self.cumulative_hazard_ = estimate.to_frame(self._label)
        self.survival_function_ = np.exp(-estimate).to_frame(self._label)
        self._estimation_method = "cumulative_hazard_"
        level = self.alpha if alpha is None else alpha
        z = norm.ppf(1.0 - level / 2.0)
        values = estimate.to_numpy()
        scale = np.divide(
            np.sqrt(variance.to_numpy()),
            values,
            out=np.zeros_like(values),
            where=values != 0,
        )
        names = ci_labels or [
            f"{self._label}_lower_{1-level:g}",
            f"{self._label}_upper_{1-level:g}",
        ]
        self.confidence_interval_ = pd.DataFrame(
            {
                names[0]: values * np.exp(-z * scale),
                names[1]: values * np.exp(z * scale),
            },
            index=estimate.index,
        )
        self.confidence_interval_cumulative_hazard_ = self.confidence_interval_
        self._cumulative_sq = variance
        return self

    def _survival(self, times):
        return np.exp(-self._cumulative_hazard(times))

    def _cumulative_hazard(self, times):
        return np.asarray(_predict_frame(self.cumulative_hazard_, times))

    def _hazard(self, times):
        raise NotImplementedError("specify a smoothing bandwidth with upstream lifelines")


class BreslowFlemingHarringtonFitter(NelsonAalenFitter):
    def __init__(self, alpha: float = 0.05, label: str = None):
        super().__init__(alpha=alpha, nelson_aalen_smoothing=True, label=label)

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label=None,
        alpha=None,
        ci_labels=None,
        weights=None,
        fit_options=None,
    ):
        if fit_options:
            raise NotImplementedError("BFH fit_options are not covered")
        selected_label = label or self._label or "BFH_estimate"
        super().fit(
            durations,
            event_observed,
            timeline,
            entry,
            selected_label,
            alpha,
            ci_labels,
            weights,
        )
        self._label = selected_label
        hazard = self.cumulative_hazard_.iloc[:, 0]
        self.survival_function_ = np.exp(-hazard).to_frame(self._label)
        self.cumulative_hazard_.columns = [self._label]
        self.confidence_interval_ = np.exp(-self.confidence_interval_)
        self.confidence_interval_survival_function_ = self.confidence_interval_
        density_ci = 1.0 - self.confidence_interval_
        density_ci.iloc[:, :] = density_ci.iloc[:, ::-1].to_numpy()
        self.confidence_interval_cumulative_density = density_ci
        self._estimation_method = "survival_function_"
        crossed = hazard.index[np.exp(-hazard) <= 0.5]
        self.median_survival_time_ = float(crossed[0]) if len(crossed) else np.inf
        return self


class _ParametricUnivariate(_Univariate):
    def _finish(self, timeline, label):
        if timeline is None:
            timeline = np.linspace(
                float(np.min(self.durations)),
                float(np.max(self.durations)),
                self.durations.size,
            )
        self.timeline = f64(timeline)
        self._label = label or self._label
        cumulative = self._cumulative_hazard(self.timeline)
        self.cumulative_hazard_ = pd.DataFrame(cumulative, index=self.timeline, columns=[self._label])
        self.survival_function_ = pd.DataFrame(np.exp(-cumulative), index=self.timeline, columns=[self._label])
        self.hazard_ = pd.DataFrame(self._hazard(self.timeline), index=self.timeline, columns=[self._label])
        self.cumulative_density_ = 1.0 - self.survival_function_
        self.density_ = self.hazard_ * self.survival_function_.to_numpy()
        self._estimation_method = "survival_function_"
        self.median_survival_time_ = float(self.percentile(0.5))
        return self

    def predict(self, times, interpolate=False):
        scalar = np.isscalar(times)
        values = np.atleast_1d(np.asarray(times, dtype=np.float64))
        result = pd.Series(self._survival(values), index=values, name=self._label)
        return float(result.iloc[0]) if scalar else result


class ExponentialFitter(_ParametricUnivariate):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop("alpha", 0.05)
        self._label = kwargs.pop("label", "Exponential_estimate")

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options: Optional[dict] = None,
    ):
        if initial_point is not None or fit_options:
            raise NotImplementedError("initial_point and fit_options are not covered")
        self.durations, self.event_observed, self.weights = _inputs(
            durations, event_observed, weights, entry
        )
        self.entry = np.zeros_like(self.durations) if entry is None else f64(entry)
        deaths = float(self.weights @ self.event_observed)
        if deaths == 0:
            raise ValueError("at least one observed event is required")
        self.lambda_ = float(self.weights @ self.durations / deaths)
        self.variance_matrix_ = pd.DataFrame(
            [[self.lambda_**2 / deaths]], index=["lambda_"], columns=["lambda_"]
        )
        self.log_likelihood_ = float(
            -deaths * np.log(self.lambda_) - self.weights @ self.durations / self.lambda_
        )
        self.AIC_ = 2.0 - 2.0 * self.log_likelihood_
        return self._finish(timeline, label)

    def _cumulative_hazard(self, times):
        return np.asarray(times) / self.lambda_

    def _hazard(self, times):
        return np.full_like(np.asarray(times), 1.0 / self.lambda_, dtype=float)

    def _survival(self, times):
        return np.exp(-self._cumulative_hazard(times))

    def percentile(self, p):
        return -self.lambda_ * np.log(p)


class WeibullFitter(_ParametricUnivariate):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop("alpha", 0.05)
        self._label = kwargs.pop("label", "Weibull_estimate")

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options: Optional[dict] = None,
    ):
        if initial_point is not None:
            raise NotImplementedError("initial_point is not covered")
        self.durations, self.event_observed, self.weights = _inputs(
            durations, event_observed, weights, entry
        )
        self.entry = np.zeros_like(self.durations) if entry is None else f64(entry)
        result = np.empty(5, dtype=np.float64)
        options = fit_options or {}
        unknown_options = set(options) - {"max_iter", "tol"}
        if unknown_options:
            raise NotImplementedError(
                f"unsupported Weibull fit_options: {sorted(unknown_options)}"
            )
        ok = lib().ml_weibull_fit(
            addr(self.durations),
            addr(self.event_observed),
            addr(self.weights),
            self.durations.size,
            addr(result),
            int(options.get("max_iter", 100)),
            float(options.get("tol", 1e-12)),
        )
        if not ok:
            raise ValueError("Weibull fitting requires positive durations and an observed event")
        if not np.all(np.isfinite(result)) or result[0] <= 0 or result[1] <= 0:
            raise RuntimeError("native Weibull fit produced an invalid result")
        self.lambda_, self.rho_ = map(float, result[:2])
        self.variance_matrix_ = pd.DataFrame(
            [[result[2], result[3]], [result[3], result[4]]],
            index=["lambda_", "rho_"],
            columns=["lambda_", "rho_"],
        )
        log_t_lambda = np.log(self.durations / self.lambda_)
        self.log_likelihood_ = float(
            np.sum(
                self.weights
                * (
                    self.event_observed
                    * (np.log(self.rho_) - np.log(self.lambda_) + (self.rho_ - 1) * log_t_lambda)
                    - np.exp(self.rho_ * log_t_lambda)
                )
            )
        )
        self.AIC_ = 4.0 - 2.0 * self.log_likelihood_
        return self._finish(timeline, label)

    def _cumulative_hazard(self, times):
        values = np.clip(np.asarray(times, dtype=float), 1e-25, np.inf)
        return (values / self.lambda_) ** self.rho_

    def _hazard(self, times):
        values = np.asarray(times, dtype=float)
        return self.rho_ / self.lambda_ * (values / self.lambda_) ** (self.rho_ - 1)

    def _survival(self, times):
        return np.exp(-self._cumulative_hazard(times))

    def percentile(self, p):
        return self.lambda_ * np.log(1.0 / p) ** (1.0 / self.rho_)


class CoxPHFitter:
    def __init__(
        self,
        baseline_estimation_method: str = "breslow",
        penalizer: Union[float, np.ndarray] = 0.0,
        strata: Optional[Union[List[str], str]] = None,
        l1_ratio: float = 0.0,
        n_baseline_knots: Optional[int] = None,
        knots: Optional[List] = None,
        breakpoints: Optional[List] = None,
        **kwargs,
    ) -> None:
        if baseline_estimation_method != "breslow":
            raise NotImplementedError("only Breslow baseline estimation is covered")
        if strata is not None:
            raise NotImplementedError("stratified Cox models are not covered")
        if l1_ratio != 0:
            raise NotImplementedError("L1 penalization is not covered")
        if any(value is not None for value in (n_baseline_knots, knots, breakpoints)):
            raise NotImplementedError("spline knots and breakpoints are not covered")
        if kwargs:
            raise NotImplementedError(f"unsupported constructor options: {sorted(kwargs)}")
        self.baseline_estimation_method = baseline_estimation_method
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def _evaluate(self, beta):
        beta = np.ascontiguousarray(beta, dtype=np.float64)
        if beta.shape != (self._d,) or not np.all(np.isfinite(beta)):
            raise ValueError("Cox coefficients must be a finite vector matching covariates")
        gradient = np.empty(self._d)
        hessian = np.empty((self._d, self._d))
        risk1 = np.empty(self._d)
        risk2 = np.empty((self._d, self._d))
        death1 = np.empty(self._d)
        death2 = np.empty((self._d, self._d))
        ll = lib().ml_cox_efron(
            addr(self._x),
            addr(self.durations),
            addr(self.event_observed),
            addr(beta),
            self._n,
            self._d,
            addr(gradient),
            addr(hessian),
            addr(risk1),
            addr(risk2),
            addr(death1),
            addr(death2),
        )
        if not np.isfinite(ll) or not np.all(np.isfinite(gradient)) or not np.all(
            np.isfinite(hessian)
        ):
            raise RuntimeError("native Cox evaluation produced non-finite values")
        penalty = np.asarray(self.penalizer, dtype=float)
        if penalty.ndim > 1 or (penalty.ndim == 1 and penalty.shape != (self._d,)):
            raise ValueError("penalizer must be scalar or match the number of covariates")
        if not np.all(np.isfinite(penalty)) or np.any(penalty < 0):
            raise ValueError("penalizer must be finite and non-negative")
        if np.any(penalty):
            ll -= 0.5 * self._n * float(np.sum(penalty * beta * beta))
            gradient -= self._n * penalty * beta
            hessian[np.diag_indices(self._d)] -= self._n * penalty
        return ll, gradient, hessian

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str] = None,
        event_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[np.ndarray] = None,
        strata: Optional[Union[str, List[str]]] = None,
        weights_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        robust: bool = False,
        batch_mode: Optional[bool] = None,
        timeline: Optional[Iterator] = None,
        formula: str = None,
        entry_col: str = None,
        fit_options: Optional[dict] = None,
    ):
        if duration_col is None:
            raise ValueError("duration_col is required")
        if any(x is not None for x in (strata, weights_col, cluster_col, formula, entry_col)):
            raise NotImplementedError("strata, weights, clusters, formulas, and delayed entry are not covered")
        if robust:
            raise NotImplementedError("robust sandwich errors are not covered")
        if batch_mode is not None:
            raise NotImplementedError("batch_mode is not covered")
        if timeline is not None:
            raise NotImplementedError("fit timeline is not covered")
        options = fit_options or {}
        unknown_options = set(options) - {"precision", "max_steps"}
        if unknown_options:
            raise NotImplementedError(
                f"unsupported Cox fit_options: {sorted(unknown_options)}"
            )
        frame = pd.DataFrame(df).copy()
        if frame.empty:
            raise ValueError("Cox fitting requires at least one observation")
        self.duration_col = duration_col
        self.event_col = event_col
        events = np.ones(len(frame), dtype=np.float64) if event_col is None else f64(frame.pop(event_col))
        times = f64(frame.pop(duration_col))
        if (
            events.shape != times.shape
            or not np.all((events == 0) | (events == 1))
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
        ):
            raise ValueError("durations must be finite and non-negative and events must be 0/1")
        x_frame = frame.astype(float)
        if not np.all(np.isfinite(x_frame.to_numpy())):
            raise ValueError("covariates must be finite numeric values")
        order = np.argsort(times, kind="stable")
        self.durations = np.ascontiguousarray(times[order])
        self.event_observed = np.ascontiguousarray(events[order])
        self._columns = x_frame.columns.copy()
        self._norm_mean = x_frame.mean(axis=0)
        centered = x_frame - self._norm_mean
        self._x = np.ascontiguousarray(centered.to_numpy()[order], dtype=np.float64)
        self._n, self._d = self._x.shape
        if self._d == 0 or np.sum(self.event_observed) == 0:
            raise ValueError("Cox fitting requires covariates and at least one observed event")
        beta = (
            np.zeros(self._d, dtype=np.float64)
            if initial_point is None
            else f64(initial_point, copy=True)
        )
        if beta.shape != (self._d,) or not np.all(np.isfinite(beta)):
            raise ValueError("initial_point must be a finite vector matching covariates")
        precision = float(options.get("precision", 1e-8))
        max_steps = int(options.get("max_steps", 50))
        ll, gradient, hessian = self._evaluate(beta)
        converged = False
        for iteration in range(max_steps):
            information = -hessian
            try:
                delta = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Cox information matrix is singular") from exc
            step = 1.0
            accepted = False
            while step >= 1.0 / 1024.0:
                candidate = np.ascontiguousarray(beta + step * delta)
                next_ll, next_gradient, next_hessian = self._evaluate(candidate)
                if next_ll >= ll:
                    beta, ll, gradient, hessian = (
                        candidate,
                        next_ll,
                        next_gradient,
                        next_hessian,
                    )
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                raise RuntimeError("Cox line search failed to improve the likelihood")
            if show_progress:
                print(f"Iteration {iteration + 1}: norm_delta={np.linalg.norm(step * delta):.3e}, log_lik={ll:.6f}")
            if np.linalg.norm(step * delta, ord=np.inf) < precision:
                converged = True
                break
        if not converged:
            raise RuntimeError(f"Cox fit did not converge in {max_steps} steps")
        self._converged = converged
        self.params_ = pd.Series(beta, index=self._columns, name="coef")
        self.hazard_ratios_ = np.exp(self.params_)
        covariance = np.linalg.inv(-hessian)
        self.variance_matrix_ = pd.DataFrame(covariance, index=self._columns, columns=self._columns)
        self.standard_errors_ = pd.Series(np.sqrt(np.diag(covariance)), index=self._columns, name="se")
        z = norm.ppf(0.975)
        self.confidence_intervals_ = pd.DataFrame(
            {
                "95% lower-bound": self.params_ - z * self.standard_errors_,
                "95% upper-bound": self.params_ + z * self.standard_errors_,
            }
        )
        self.log_likelihood_ = float(ll)
        self.AIC_partial_ = 2 * self._d - 2 * self.log_likelihood_
        self._compute_baseline()
        self.concordance_index_ = self._concordance(
            self.durations, self.event_observed, self._predict_centered(self._x)
        )
        self.summary = pd.DataFrame(
            {
                "coef": self.params_,
                "exp(coef)": self.hazard_ratios_,
                "se(coef)": self.standard_errors_,
                "coef lower 95%": self.confidence_intervals_.iloc[:, 0],
                "coef upper 95%": self.confidence_intervals_.iloc[:, 1],
                "z": self.params_ / self.standard_errors_,
            }
        )
        return self

    def _predict_centered(self, centered):
        matrix = np.ascontiguousarray(centered, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self._d:
            raise ValueError("prediction data has the wrong number of covariates")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("prediction covariates must be finite")
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        result = np.empty(matrix.shape[0], dtype=np.float64)
        beta = np.ascontiguousarray(self.params_.to_numpy())
        lib().ml_predict_log_hazard(addr(matrix), addr(beta), addr(result), matrix.shape[0], matrix.shape[1])
        return result

    def _prepare_prediction(self, x):
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        frame = pd.DataFrame(x)
        if all(column in frame.columns for column in self._columns):
            frame = frame.loc[:, self._columns]
        elif frame.shape[1] != len(self._columns):
            raise ValueError("prediction data has the wrong number of covariates")
        return frame, np.ascontiguousarray(frame.to_numpy(dtype=float) - self._norm_mean.to_numpy())

    def predict_log_partial_hazard(self, X):
        frame, centered = self._prepare_prediction(X)
        return pd.Series(self._predict_centered(centered), index=frame.index)

    def predict_partial_hazard(self, X):
        return np.exp(self.predict_log_partial_hazard(X))

    def _compute_baseline(self):
        eta = self._predict_centered(self._x)
        hazard = np.exp(eta)
        risk = np.cumsum(hazard[::-1])[::-1]
        unique, first, counts = np.unique(self.durations, return_index=True, return_counts=True)
        deaths = np.add.reduceat(self.event_observed, first)
        increments = deaths / risk[first]
        baseline = pd.Series(increments, index=unique)
        self.baseline_hazard_ = baseline.to_frame("baseline hazard")
        self.baseline_cumulative_hazard_ = baseline.cumsum().to_frame("baseline cumulative hazard")
        self.baseline_survival_ = np.exp(-self.baseline_cumulative_hazard_).rename(
            columns={"baseline cumulative hazard": "baseline survival"}
        )

    def predict_cumulative_hazard(self, X, times=None, conditional_after=None):
        if conditional_after is not None:
            raise NotImplementedError("conditional_after is not covered")
        partial = self.predict_partial_hazard(X)
        baseline = self.baseline_cumulative_hazard_.iloc[:, 0]
        if times is not None:
            query = np.asarray(times, dtype=float)
            baseline = (
                baseline.reindex(baseline.index.union(query))
                .interpolate("index")
                .reindex(query)
                .fillna(0.0)
            )
        values = np.outer(baseline.to_numpy(), partial.to_numpy())
        return pd.DataFrame(values, index=baseline.index, columns=partial.index)

    def predict_survival_function(self, X, times=None, conditional_after=None):
        return np.exp(-self.predict_cumulative_hazard(X, times, conditional_after))

    def predict_percentile(self, X, p=0.5, conditional_after=None):
        survival = self.predict_survival_function(X, conditional_after=conditional_after)
        result = {}
        for column in survival:
            crossed = survival.index[survival[column] <= p]
            result[column] = float(crossed[0]) if len(crossed) else np.inf
        return pd.Series(result)

    def predict_median(self, X, conditional_after=None):
        return self.predict_percentile(X, p=0.5, conditional_after=conditional_after)

    def predict_expectation(self, X, conditional_after=None):
        survival = self.predict_survival_function(X, conditional_after=conditional_after)
        return pd.Series(
            np.trapezoid(survival.to_numpy(), survival.index.to_numpy(), axis=0),
            index=survival.columns,
        )

    @staticmethod
    def _concordance(times, events, log_hazard):
        concordant = 0.0
        comparable = 0
        _, ranks = np.unique(log_hazard, return_inverse=True)
        tree = np.zeros(int(ranks.max()) + 2, dtype=np.int64)

        def add(rank):
            index = int(rank) + 1
            while index < len(tree):
                tree[index] += 1
                index += index & -index

        def prefix(rank):
            total = 0
            index = int(rank) + 1
            while index:
                total += int(tree[index])
                index -= index & -index
            return total

        order = np.argsort(times)[::-1]
        prior = 0
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and times[order[end]] == times[order[start]]:
                end += 1
            for position in range(start, end):
                row = order[position]
                if events[row]:
                    below = prefix(ranks[row] - 1) if ranks[row] else 0
                    equal = prefix(ranks[row]) - below
                    concordant += below + 0.5 * equal
                    comparable += prior
            for position in range(start, end):
                add(ranks[order[position]])
                prior += 1
            start = end
        return concordant / comparable if comparable else np.nan

    def score(self, df: pd.DataFrame, scoring_method: str = "log_likelihood"):
        if scoring_method not in ("log_likelihood", "concordance_index"):
            raise NotImplementedError("supported scoring methods are log_likelihood and concordance_index")
        frame = pd.DataFrame(df).copy()
        events = (
            np.ones(len(frame), dtype=np.float64)
            if self.event_col is None
            else f64(frame.pop(self.event_col))
        )
        times = f64(frame.pop(self.duration_col))
        _, centered = self._prepare_prediction(frame)
        log_hazard = self._predict_centered(centered)
        if scoring_method == "concordance_index":
            return self._concordance(times, events, log_hazard)
        order = np.argsort(times, kind="stable")
        saved = self._x, self.durations, self.event_observed, self._n
        self._x = np.ascontiguousarray(centered[order])
        self.durations = np.ascontiguousarray(times[order])
        self.event_observed = np.ascontiguousarray(events[order])
        self._n = len(times)
        try:
            value = self._evaluate(np.ascontiguousarray(self.params_.to_numpy()))[0]
        finally:
            self._x, self.durations, self.event_observed, self._n = saved
        return value / len(times)
