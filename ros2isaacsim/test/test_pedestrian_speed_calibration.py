import math

import pytest

from ros2isaacsim.pedestrian_speed_calibration import (
    SpeedTrial,
    estimate_projected_root_speed,
    fit_walk_blend_power_law,
    parse_speed_levels,
    root_motion_diagnostics,
    summarize_speed_samples,
)


def test_parse_speed_levels_requires_strictly_increasing_values():
    assert parse_speed_levels("0.2, 0.4,0.7") == (0.2, 0.4, 0.7)
    with pytest.raises(Exception):
        parse_speed_levels("0.4,0.2")


def test_speed_summary_trims_single_outlier():
    median, deviation, count = summarize_speed_samples(
        [0.39, 0.40, 0.41, 0.40, 0.39, 0.40, 0.41, 0.40, 0.39, 4.0]
    )
    assert median == pytest.approx(0.40)
    assert deviation < 0.02
    assert count == 8


def test_projected_speed_uses_cycle_average_position_slope():
    samples = [
        (index * 0.1, 1.2 * index * 0.1 + 0.02 * math.sin(index), 2.0)
        for index in range(40)
    ]
    speed, residual, count = estimate_projected_root_speed(samples, (1.0, 0.0))
    assert speed == pytest.approx(1.2, abs=0.005)
    assert residual < 0.02
    assert count == 40


def test_root_motion_diagnostics_exposes_lateral_drift():
    samples = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.2), (2.0, 2.0, 0.4)]
    path_speed, lateral_speed = root_motion_diagnostics(samples, (1.0, 0.0))
    assert path_speed == pytest.approx(math.hypot(1.0, 0.2))
    assert lateral_speed == pytest.approx(0.2)


def test_power_law_fit_recovers_known_mapping():
    full_speed = 1.12
    exponent = 1.7
    trials = [
        SpeedTrial(
            commanded_speed_mps=blend,
            walk_blend=blend,
            measured_speed_mps=full_speed * blend**exponent,
            sample_count=30,
            position_fit_rmse_m=0.01,
        )
        for blend in (0.25, 0.40, 0.60, 0.80)
    ]
    fit = fit_walk_blend_power_law(trials)
    assert fit.full_speed_mps == pytest.approx(full_speed)
    assert fit.speed_exponent == pytest.approx(exponent)
    assert fit.rmse_mps < 1e-10
    assert fit.max_abs_error_mps < 1e-10


def test_power_law_fit_ignores_saturated_blend():
    trials = [
        SpeedTrial(0.2, 0.3, 0.2, 20, 0.01),
        SpeedTrial(0.4, 0.6, 0.4, 20, 0.01),
        SpeedTrial(1.2, 1.0, 9.0, 20, 0.01),
    ]
    fit = fit_walk_blend_power_law(trials)
    assert math.isfinite(fit.full_speed_mps)
    assert fit.max_abs_error_mps < 1e-9
