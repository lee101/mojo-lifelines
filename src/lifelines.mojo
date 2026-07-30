"""Numerical kernels for right-censored survival models."""

from std.math import exp, log, pow
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ptr(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@export("ml_event_table")
def ml_event_table(
    durations_addr: Int,
    events_addr: Int,
    weights_addr: Int,
    n: Int,
    times_addr: Int,
    removed_addr: Int,
    observed_addr: Int,
    censored_addr: Int,
    at_risk_addr: Int,
    survival_addr: Int,
    greenwood_addr: Int,
    na_smooth_addr: Int,
    na_discrete_addr: Int,
    na_smooth_variance_addr: Int,
    na_discrete_variance_addr: Int,
) abi("C") -> Int:
    var durations = ptr(durations_addr)
    var events = ptr(events_addr)
    var weights = ptr(weights_addr)
    var times = ptr(times_addr)
    var removed = ptr(removed_addr)
    var observed = ptr(observed_addr)
    var censored = ptr(censored_addr)
    var at_risk = ptr(at_risk_addr)
    var survival = ptr(survival_addr)
    var greenwood = ptr(greenwood_addr)
    var na_smooth = ptr(na_smooth_addr)
    var na_discrete = ptr(na_discrete_addr)
    var na_smooth_variance = ptr(na_smooth_variance_addr)
    var na_discrete_variance = ptr(na_discrete_variance_addr)

    comptime W = simd_width_of[DType.float64]()
    var remaining = 0.0
    var sum_i = 0
    while sum_i + W <= n:
        remaining += weights.load[width=W](sum_i).reduce_add()
        sum_i += W
    while sum_i < n:
        remaining += weights[sum_i]
        sum_i += 1

    var s = 1.0
    var g = 0.0
    var hs = 0.0
    var hd = 0.0
    var hs_variance = 0.0
    var hd_variance = 0.0
    var i = 0
    var row = 0
    while i < n:
        var t = durations[i]
        var r = 0.0
        var d = 0.0
        var upper = i + 1
        while upper < n and durations[upper] == t:
            upper += 1
        while i + W <= upper:
            var weight_values = weights.load[width=W](i)
            r += weight_values.reduce_add()
            d += (weight_values * events.load[width=W](i)).reduce_add()
            i += W
        while i < upper:
            r += weights[i]
            d += weights[i] * events[i]
            i += 1

        times[row] = t
        removed[row] = r
        observed[row] = d
        censored[row] = r - d
        at_risk[row] = remaining
        if d > 0.0 and remaining > 0.0:
            s *= 1.0 - d / remaining
            if remaining > d:
                g += d / (remaining * (remaining - d))
            hd += d / remaining
            hd_variance += (1.0 - d / remaining) * d / (remaining * remaining)
            var j = 0
            while Float64(j) < d:
                var denominator = remaining - Float64(j)
                hs += 1.0 / denominator
                hs_variance += 1.0 / (denominator * denominator)
                j += 1
        survival[row] = s
        greenwood[row] = g
        na_smooth[row] = hs
        na_discrete[row] = hd
        na_smooth_variance[row] = hs_variance
        na_discrete_variance[row] = hd_variance
        remaining -= r
        row += 1
    return row


@export("ml_weibull_fit")
def ml_weibull_fit(
    durations_addr: Int,
    events_addr: Int,
    weights_addr: Int,
    n: Int,
    result_addr: Int,
    max_iter: Int,
    tolerance: Float64,
) abi("C") -> Int:
    var durations = ptr(durations_addr)
    var events = ptr(events_addr)
    var weights = ptr(weights_addr)
    var result = ptr(result_addr)
    var deaths = 0.0
    var event_log_sum = 0.0
    for i in range(n):
        if durations[i] <= 0.0:
            return 0
        deaths += weights[i] * events[i]
        event_log_sum += weights[i] * events[i] * log(durations[i])
    if deaths <= 0.0:
        return 0

    var rho = 1.0
    for _ in range(max_iter):
        var scale = 0.0
        var scale_log = 0.0
        var scale_log2 = 0.0
        for i in range(n):
            var lt = log(durations[i])
            var q = weights[i] * exp(rho * lt)
            scale += q
            scale_log += q * lt
            scale_log2 += q * lt * lt
        var mean_log = scale_log / scale
        var variance_log = scale_log2 / scale - mean_log * mean_log
        var score = deaths / rho + event_log_sum - deaths * mean_log
        var derivative = -deaths / (rho * rho) - deaths * variance_log
        var step = score / derivative
        var next_rho = rho - step
        var halves = 0
        while next_rho <= 0.0 and halves < 30:
            step *= 0.5
            next_rho = rho - step
            halves += 1
        rho = next_rho
        if abs(step) <= tolerance * (1.0 + rho):
            break

    var power_sum = 0.0
    for i in range(n):
        power_sum += weights[i] * exp(rho * log(durations[i]))
    var lambda_ = exp(log(power_sum / deaths) / rho)
    result[0] = lambda_
    result[1] = rho

    var h_ll = -deaths * rho * rho / (lambda_ * lambda_)
    var h_lr = 0.0
    var h_rr = -deaths / (rho * rho)
    for i in range(n):
        var centered_log = log(durations[i]) - log(lambda_)
        var h = weights[i] * exp(rho * centered_log)
        h_lr += rho * h * centered_log / lambda_
        h_rr -= h * centered_log * centered_log
    var info_ll = -h_ll
    var info_lr = -h_lr
    var info_rr = -h_rr
    var determinant = info_ll * info_rr - info_lr * info_lr
    result[2] = info_rr / determinant
    result[3] = -info_lr / determinant
    result[4] = info_ll / determinant
    return 1


@export("ml_cox_efron")
def ml_cox_efron(
    x_addr: Int,
    times_addr: Int,
    events_addr: Int,
    beta_addr: Int,
    n: Int,
    d: Int,
    gradient_addr: Int,
    hessian_addr: Int,
    risk1_addr: Int,
    risk2_addr: Int,
    death1_addr: Int,
    death2_addr: Int,
) abi("C") -> Float64:
    var x = ptr(x_addr)
    var times = ptr(times_addr)
    var events = ptr(events_addr)
    var beta = ptr(beta_addr)
    var gradient = ptr(gradient_addr)
    var hessian = ptr(hessian_addr)
    var risk1 = ptr(risk1_addr)
    var risk2 = ptr(risk2_addr)
    var death1 = ptr(death1_addr)
    var death2 = ptr(death2_addr)

    for j in range(d):
        gradient[j] = 0.0
        risk1[j] = 0.0
        death1[j] = 0.0
    for j in range(d * d):
        hessian[j] = 0.0
        risk2[j] = 0.0
        death2[j] = 0.0

    var risk0 = 0.0
    var log_likelihood = 0.0
    var upper = n
    while upper > 0:
        var lower = upper - 1
        var t = times[lower]
        while lower > 0 and times[lower - 1] == t:
            lower -= 1

        var death0 = 0.0
        var death_count = 0
        for r in range(lower, upper):
            var eta = 0.0
            for j in range(d):
                eta += x[r * d + j] * beta[j]
            var hazard = exp(eta)
            risk0 += hazard
            for j in range(d):
                var xj = x[r * d + j]
                risk1[j] += hazard * xj
                for k in range(d):
                    risk2[j * d + k] += hazard * xj * x[r * d + k]
            if events[r] != 0.0:
                death_count += 1
                death0 += hazard
                log_likelihood += eta
                for j in range(d):
                    var xj = x[r * d + j]
                    gradient[j] += xj
                    death1[j] += hazard * xj
                    for k in range(d):
                        death2[j * d + k] += hazard * xj * x[r * d + k]

        if death_count > 0:
            for tie in range(death_count):
                var fraction = Float64(tie) / Float64(death_count)
                var denominator = risk0 - fraction * death0
                log_likelihood -= log(denominator)
                for j in range(d):
                    var a = (risk1[j] - fraction * death1[j]) / denominator
                    gradient[j] -= a
                    for k in range(d):
                        var b = (risk2[j * d + k] - fraction * death2[j * d + k]) / denominator
                        var ak = (risk1[k] - fraction * death1[k]) / denominator
                        hessian[j * d + k] -= b - a * ak

        for j in range(d):
            death1[j] = 0.0
        for j in range(d * d):
            death2[j] = 0.0
        upper = lower
    return log_likelihood


@export("ml_predict_log_hazard")
def ml_predict_log_hazard(
    x_addr: Int, beta_addr: Int, result_addr: Int, n: Int, d: Int
) abi("C"):
    var x = ptr(x_addr)
    var beta = ptr(beta_addr)
    var result = ptr(result_addr)
    for i in range(n):
        var value = 0.0
        for j in range(d):
            value += x[i * d + j] * beta[j]
        result[i] = value
