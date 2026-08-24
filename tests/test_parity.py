"""Numerical and behavioural parity with lifelines on right-censored data."""

import inspect

import numpy as np
import pandas as pd
import pytest

import lifelines as upstream
import mojo_lifelines as mojo


@pytest.fixture
def tied_data():
    durations = np.array([5, 6, 6, 2, 4, 4, 1, 8, 3, 3, 7], dtype=float)
    events = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    return durations, events


@pytest.fixture(scope="module")
def cox_data():
    rng = np.random.default_rng(42)
    n = 600
    x = rng.normal(size=(n, 4))
    beta = np.array([0.8, -0.45, 0.25, 0.0])
    event_time = rng.exponential(scale=1.4, size=n) / np.exp(x @ beta)
    censor_time = rng.exponential(scale=2.8, size=n)
    frame = pd.DataFrame(x, columns=["age", "marker", "dose", "noise"])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = event_time <= censor_time
    return frame


def test_public_signatures_match():
    for name in [
        "KaplanMeierFitter",
        "NelsonAalenFitter",
        "BreslowFlemingHarringtonFitter",
        "ExponentialFitter",
        "WeibullFitter",
        "CoxPHFitter",
    ]:
        assert list(inspect.signature(getattr(mojo, name).fit).parameters) == list(
            inspect.signature(getattr(upstream, name).fit).parameters
        )


def test_kaplan_meier_estimate_and_event_table(tied_data):
    durations, events = tied_data
    ours = mojo.KaplanMeierFitter().fit(durations, events)
    theirs = upstream.KaplanMeierFitter().fit(durations, events)
    assert np.allclose(ours.survival_function_, theirs.survival_function_)
    assert np.allclose(ours.event_table, theirs.event_table)
    assert ours.median_survival_time_ == theirs.median_survival_time_


def test_kaplan_meier_custom_timeline(tied_data):
    durations, events = tied_data
    timeline = np.linspace(0, 8, 33)
    ours = mojo.KaplanMeierFitter().fit(durations, events, timeline=timeline)
    theirs = upstream.KaplanMeierFitter().fit(durations, events, timeline=timeline)
    assert np.allclose(ours.survival_function_, theirs.survival_function_)
    assert np.allclose(ours.cumulative_density_, theirs.cumulative_density_)


def test_kaplan_meier_confidence_interval(tied_data):
    durations, events = tied_data
    ours = mojo.KaplanMeierFitter(alpha=0.1).fit(durations, events)
    theirs = upstream.KaplanMeierFitter(alpha=0.1).fit(durations, events)
    assert np.allclose(ours.confidence_interval_, theirs.confidence_interval_)


def test_kaplan_meier_integer_weights(tied_data):
    durations, events = tied_data
    weights = np.array([1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1])
    ours = mojo.KaplanMeierFitter().fit(durations, events, weights=weights)
    theirs = upstream.KaplanMeierFitter().fit(durations, events, weights=weights)
    assert np.allclose(ours.survival_function_, theirs.survival_function_)
    assert np.allclose(ours.event_table, theirs.event_table)


@pytest.mark.parametrize("smoothing", [True, False])
def test_nelson_aalen(tied_data, smoothing):
    durations, events = tied_data
    ours = mojo.NelsonAalenFitter(nelson_aalen_smoothing=smoothing).fit(durations, events)
    theirs = upstream.NelsonAalenFitter(nelson_aalen_smoothing=smoothing).fit(durations, events)
    assert np.allclose(ours.cumulative_hazard_, theirs.cumulative_hazard_)
    assert np.allclose(ours.confidence_interval_, theirs.confidence_interval_)
    points = np.array([0.0, 2.5, 4.0, 9.0])
    assert np.allclose(ours.predict(points), theirs.predict(points))


@pytest.mark.parametrize("smoothing", [True, False])
def test_nelson_aalen_simd_tail(smoothing):
    durations = np.arange(1.0, 20.0)
    events = np.array(
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        dtype=bool,
    )
    weights = np.array(
        [1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 2, 1, 3, 1],
        dtype=float,
    )
    ours = mojo.NelsonAalenFitter(nelson_aalen_smoothing=smoothing).fit(
        durations, events, weights=weights
    )
    theirs = upstream.NelsonAalenFitter(nelson_aalen_smoothing=smoothing).fit(
        durations, events, weights=weights
    )
    assert np.allclose(ours.event_table, theirs.event_table)
    assert np.allclose(ours.cumulative_hazard_, theirs.cumulative_hazard_)
    assert np.allclose(ours.confidence_interval_, theirs.confidence_interval_)


def test_dense_integer_event_scan():
    n = 4097
    durations = (np.arange(n) * 17 % 53).astype(float)
    events = np.arange(n) % 3 != 0
    weights = (np.arange(n) % 4 + 1).astype(float)

    ours_km = mojo.KaplanMeierFitter().fit(durations, events, weights=weights)
    theirs_km = upstream.KaplanMeierFitter().fit(durations, events, weights=weights)
    assert np.allclose(ours_km.event_table, theirs_km.event_table)
    assert np.allclose(ours_km.survival_function_, theirs_km.survival_function_)

    ours_na = mojo.NelsonAalenFitter().fit(durations, events, weights=weights)
    theirs_na = upstream.NelsonAalenFitter().fit(durations, events, weights=weights)
    assert np.allclose(ours_na.event_table, theirs_na.event_table)
    assert np.allclose(ours_na.cumulative_hazard_, theirs_na.cumulative_hazard_)


def test_breslow_fleming_harrington(tied_data):
    durations, events = tied_data
    ours = mojo.BreslowFlemingHarringtonFitter().fit(durations, events)
    theirs = upstream.BreslowFlemingHarringtonFitter().fit(durations, events)
    assert np.allclose(ours.survival_function_, theirs.survival_function_)
    assert list(ours.survival_function_.columns) == list(theirs.survival_function_.columns)
    assert np.allclose(ours.confidence_interval_, theirs.confidence_interval_)
    assert ours.median_survival_time_ == theirs.median_survival_time_


def test_exponential_fit(tied_data):
    durations, events = tied_data
    ours = mojo.ExponentialFitter().fit(durations, events)
    theirs = upstream.ExponentialFitter().fit(durations, events)
    assert ours.lambda_ == pytest.approx(theirs.lambda_, rel=2e-8)
    assert ours.log_likelihood_ == pytest.approx(theirs.log_likelihood_, abs=1e-9)
    assert np.allclose(ours.survival_function_, theirs.survival_function_, atol=2e-9)
    assert np.allclose(ours.variance_matrix_, theirs.variance_matrix_, rtol=2e-8)
    assert ours.median_survival_time_ == pytest.approx(theirs.median_survival_time_, rel=2e-8)


def test_exponential_weighted_fit(tied_data):
    durations, events = tied_data
    weights = np.arange(1, len(durations) + 1)
    ours = mojo.ExponentialFitter().fit(durations, events, weights=weights)
    theirs = upstream.ExponentialFitter().fit(durations, events, weights=weights)
    assert ours.lambda_ == pytest.approx(theirs.lambda_, rel=2e-8)
    points = np.linspace(0.1, 10, 20)
    assert np.allclose(
        ours.cumulative_hazard_at_times(points),
        theirs.cumulative_hazard_at_times(points),
        rtol=2e-8,
    )


def test_weibull_fit(tied_data):
    durations, events = tied_data
    ours = mojo.WeibullFitter().fit(durations, events)
    theirs = upstream.WeibullFitter().fit(durations, events)
    assert ours.lambda_ == pytest.approx(theirs.lambda_, rel=2e-5)
    assert ours.rho_ == pytest.approx(theirs.rho_, rel=2e-5)
    assert ours.log_likelihood_ == pytest.approx(theirs.log_likelihood_, abs=1e-7)
    assert np.allclose(ours.survival_function_, theirs.survival_function_, atol=2e-5)
    assert np.allclose(ours.variance_matrix_, theirs.variance_matrix_, rtol=2e-4)


def test_weibull_predictions(tied_data):
    durations, events = tied_data
    ours = mojo.WeibullFitter().fit(durations, events)
    theirs = upstream.WeibullFitter().fit(durations, events)
    points = np.linspace(0.25, 12, 30)
    assert np.allclose(
        ours.survival_function_at_times(points),
        theirs.survival_function_at_times(points),
        rtol=1e-4,
    )
    assert np.allclose(
        ours.hazard_at_times(points), theirs.hazard_at_times(points), rtol=1e-4
    )
    assert ours.percentile(0.25) == pytest.approx(theirs.percentile(0.25), rel=3e-5)


def test_cox_coefficients_and_likelihood(cox_data):
    ours = mojo.CoxPHFitter().fit(cox_data, "time", "event")
    theirs = upstream.CoxPHFitter().fit(cox_data, "time", "event")
    assert np.allclose(ours.params_, theirs.params_, atol=2e-5)
    assert ours.log_likelihood_ == pytest.approx(theirs.log_likelihood_, abs=2e-6)
    assert np.allclose(ours.variance_matrix_, theirs.variance_matrix_, atol=2e-6)


def test_cox_efron_ties():
    frame = pd.DataFrame(
        {
            "x1": [-1.0, -0.5, 0.1, 0.7, 1.2, -0.3, 0.8, 1.5],
            "x2": [0.2, 1.0, -0.7, 0.3, -1.2, 1.3, 0.1, -0.4],
            "time": [1, 1, 2, 2, 2, 3, 4, 4],
            "event": [1, 1, 1, 0, 1, 1, 0, 1],
        }
    )
    ours = mojo.CoxPHFitter().fit(frame, "time", "event")
    theirs = upstream.CoxPHFitter().fit(frame, "time", "event")
    assert np.allclose(ours.params_, theirs.params_, atol=2e-5)
    assert ours.log_likelihood_ == pytest.approx(theirs.log_likelihood_, abs=2e-6)


def test_cox_partial_hazard(cox_data):
    ours = mojo.CoxPHFitter().fit(cox_data, "time", "event")
    theirs = upstream.CoxPHFitter().fit(cox_data, "time", "event")
    covariates = cox_data.drop(columns=["time", "event"]).iloc[:25]
    assert np.allclose(
        ours.predict_log_partial_hazard(covariates),
        theirs.predict_log_partial_hazard(covariates),
        atol=3e-5,
    )
    assert np.allclose(
        ours.predict_partial_hazard(covariates),
        theirs.predict_partial_hazard(covariates),
        rtol=3e-5,
    )


def test_cox_baseline_and_survival(cox_data):
    ours = mojo.CoxPHFitter().fit(cox_data, "time", "event")
    theirs = upstream.CoxPHFitter().fit(cox_data, "time", "event")
    assert np.allclose(
        ours.baseline_cumulative_hazard_,
        theirs.baseline_cumulative_hazard_,
        rtol=3e-5,
        atol=2e-7,
    )
    covariates = cox_data.drop(columns=["time", "event"]).iloc[:4]
    times = np.linspace(0.1, 2.0, 20)
    assert np.allclose(
        ours.predict_survival_function(covariates, times=times),
        theirs.predict_survival_function(covariates, times=times),
        rtol=1e-4,
        atol=2e-7,
    )


def test_cox_summary_and_concordance(cox_data):
    ours = mojo.CoxPHFitter().fit(cox_data, "time", "event")
    theirs = upstream.CoxPHFitter().fit(cox_data, "time", "event")
    assert list(ours.summary.index) == list(theirs.summary.index)
    assert np.all(np.isfinite(ours.standard_errors_))
    assert ours.concordance_index_ == pytest.approx(theirs.concordance_index_, abs=1e-12)
    assert ours.score(cox_data, "concordance_index") == pytest.approx(
        theirs.score(cox_data, "concordance_index"), abs=1e-12
    )
    assert ours.score(cox_data, "log_likelihood") == pytest.approx(
        theirs.score(cox_data, "log_likelihood"), abs=2e-8
    )


def test_unsupported_delayed_entry_is_explicit(tied_data):
    durations, events = tied_data
    with pytest.raises(NotImplementedError, match="delayed entry"):
        mojo.KaplanMeierFitter().fit(durations, events, entry=np.ones(len(durations)))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0, np.nan], "finite"),
        ([1.0, np.inf], "finite"),
    ],
)
def test_invalid_weights_are_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        mojo.KaplanMeierFitter().fit([1, 2], weights=values)


def test_integer_narrowing_is_rejected():
    with pytest.raises(ValueError, match="exact float64"):
        mojo.KaplanMeierFitter().fit(np.array([1, 2**53 + 1], dtype=np.int64))


def test_unsupported_options_are_explicit(tied_data, cox_data):
    durations, events = tied_data
    with pytest.raises(NotImplementedError, match="fit_options"):
        mojo.KaplanMeierFitter().fit(durations, events, fit_options={"unused": True})
    with pytest.raises(NotImplementedError, match="initial_point"):
        mojo.ExponentialFitter().fit(durations, events, initial_point=[1.0])
    with pytest.raises(NotImplementedError, match="batch_mode"):
        mojo.CoxPHFitter().fit(cox_data, "time", "event", batch_mode=True)


def test_cox_rejects_invalid_native_inputs(cox_data):
    bad_event = cox_data.copy()
    bad_event["event"] = bad_event["event"].astype(int)
    bad_event.loc[0, "event"] = 2
    with pytest.raises(ValueError, match="events"):
        mojo.CoxPHFitter().fit(bad_event, "time", "event")

    bad_time = cox_data.copy()
    bad_time.loc[0, "time"] = np.nan
    with pytest.raises(ValueError, match="durations"):
        mojo.CoxPHFitter().fit(bad_time, "time", "event")

    with pytest.raises(ValueError, match="initial_point"):
        mojo.CoxPHFitter().fit(
            cox_data, "time", "event", initial_point=np.zeros(2)
        )


def test_empty_cox_prediction_is_safe(cox_data):
    fitted = mojo.CoxPHFitter().fit(cox_data, "time", "event")
    empty = cox_data.drop(columns=["time", "event"]).iloc[:0]
    result = fitted.predict_partial_hazard(empty)
    assert result.empty
