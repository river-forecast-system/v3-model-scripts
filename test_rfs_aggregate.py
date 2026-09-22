"""
Checks for ``rfs_aggregate.fold_year`` and the accumulators around it.

The kernel folds one year into five products in a single pass, so the thing worth testing is that each of them is
what the obvious, slow, one-product-at-a-time numpy version gives -- and, for the rounding, that it is bit for bit
what the two codecs the rest of the build uses give. Run with ``python test_rfs_aggregate.py`` or under pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numcodecs import BitRound
from river_route.router.writers import bitround

import rfs_aggregate
import rfs_spec


def hours_of(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left')


def synthetic(n_rivers: int, dates: pd.DatetimeIndex, seed: int = 0, scale: float = 5_000.0) -> np.ndarray:
    """A (river, hour) buffer with a seasonal shape and noise, so daily and monthly means differ from each other."""
    rng = np.random.default_rng(seed)
    hours = np.arange(len(dates), dtype=np.float32)
    season = 1.0 + 0.8 * np.sin(2 * np.pi * hours / len(dates))
    daily = 1.0 + 0.3 * np.sin(2 * np.pi * hours / 24)
    base = rng.random((n_rivers, 1), dtype=np.float32) * scale
    noise = rng.random((n_rivers, len(dates)), dtype=np.float32)
    return np.ascontiguousarray(base * season * daily * (0.5 + noise), dtype=np.float32)


def reference(hourly: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The five products, computed one at a time with pandas labels and numpy reductions."""
    keepbits = rfs_spec.KEEP_BITS
    rounded = bitround(hourly.astype(np.float32, copy=True), keepbits)
    day_starts = np.flatnonzero(np.r_[True, dates.to_period('D')[1:] != dates.to_period('D')[:-1]])
    day_months = dates[day_starts].to_period('M')
    month_starts = np.flatnonzero(np.r_[True, day_months[1:] != day_months[:-1]])

    day_counts = np.diff(np.r_[day_starts, len(dates)])
    day_sums = np.add.reduceat(rounded, day_starts, axis=1, dtype=np.float64)
    daily = bitround((day_sums / day_counts).astype(np.float32), keepbits)

    month_hours = np.add.reduceat(day_counts, month_starts)
    month_sums = np.add.reduceat(day_sums, month_starts, axis=1)
    monthly = bitround((month_sums / month_hours).astype(np.float32), keepbits)

    yearly = bitround((day_sums.sum(axis=1) / len(dates)).astype(np.float32), keepbits)
    return {
        'rounded': rounded,
        'daily': daily,
        'monthly': monthly,
        'yearly': yearly,
        'max_hourly': rounded.max(axis=1),
        'max_daily': daily.max(axis=1),
    }


def fold_one_year(hourly: np.ndarray, dates: pd.DatetimeIndex) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fold a single year through a RegionAccumulators and return its arrays and the rounded buffer."""
    accumulators = rfs_aggregate.RegionAccumulators(hourly.shape[0], [dates[0].year])
    accumulators.fold(dates, hourly)
    return {
        'daily': accumulators.daily,
        'monthly': accumulators.monthly,
        'yearly': accumulators.yearly[:, 0],
        'max_hourly': accumulators.max_hourly[:, 0],
        'max_daily': accumulators.max_daily[:, 0],
    }, hourly


def assert_agrees_with_float64(got: np.ndarray, expect: np.ndarray, what: str) -> None:
    """
    The kernel agrees with a float64 reduction to within one step of the keepbits grid, and almost always exactly.

    It cannot be bit for bit everywhere: the kernel cascades float32 sums where the reference reduces in float64,
    which is deliberate -- the cascade keeps the longest chain of additions at about 67 rather than 8,784, so the
    error stays around 3e-07 against a grid whose step is 6.1e-05. Measured over 512 rivers of a leap year, daily
    and yearly come out bit identical and 9 monthly values in 6,144 land one step away, so the bound below is what
    that costs rather than a tolerance chosen to make the test pass.
    """
    got, expect = np.asarray(got), np.asarray(expect)
    steps = np.abs(got.view(np.int32).astype(np.int64) - expect.view(np.int32).astype(np.int64))
    steps >>= 23 - rfs_spec.KEEP_BITS  # the low bits are zero on the grid, so this is the distance along it
    assert steps.max() <= 1, f'{what} is {steps.max()} grid steps from the float64 reduction'
    exact = (steps == 0).mean()
    assert exact >= 0.99, f'only {exact:.2%} of {what} matches the float64 reduction exactly'


def test_rounding_matches_both_codecs() -> None:
    """The kernel's rounding is the one the rest of the build uses, to the bit, including at a tie."""
    rng = np.random.default_rng(7)
    values = np.concatenate([
        rng.random(50_000).astype(np.float32) * 1e5,
        np.array([0.0, 1e-8, 1.5, 3.14159, 65504.0, 2.0**-30, 1e-30], dtype=np.float32),
        # exact ties: a value sitting halfway between two points of the keepbits grid, which must round to even
        (np.arange(1, 2000, dtype=np.uint32) << np.uint32(23 - rfs_spec.KEEP_BITS - 1) | np.uint32(0x3F000000)).view(
            np.float32
        ),
    ]).astype(np.float32)
    dates = hours_of(2001)
    tiled = np.resize(values, (1, len(dates))).astype(np.float32)
    buffer = tiled.copy()
    fold_one_year(buffer, dates)  # rounds the buffer in place
    expected = bitround(tiled.copy(), rfs_spec.KEEP_BITS)
    codec = np.asarray(BitRound(keepbits=rfs_spec.KEEP_BITS).encode(tiled.copy())).view(np.float32)
    assert np.array_equal(buffer.view(np.uint32), expected.view(np.uint32)), 'differs from river-route bitround'
    assert np.array_equal(buffer.view(np.uint32), codec.view(np.uint32)), 'differs from numcodecs BitRound'


def test_float32_year_matches_numpy() -> None:
    """Every product of a leap year folds to exactly what the one-at-a-time numpy version gives."""
    dates = hours_of(2020)  # a leap year, so February is 29 days and the yearly denominator is 8,784 hours
    hourly = synthetic(64, dates, seed=1)
    expect = reference(hourly, dates)
    got, rounded = fold_one_year(hourly.copy(), dates)

    assert np.array_equal(rounded.view(np.uint32), expect['rounded'].view(np.uint32)), 'buffer not rounded in place'
    assert got['daily'].shape == (64, 366) and got['monthly'].shape == (64, 12)
    # the hourly maximum is a value picked out of the buffer, never summed, so it is exact whatever the sums do
    assert np.array_equal(got['max_hourly'].view(np.uint32), expect['max_hourly'].view(np.uint32))
    for name in ('daily', 'monthly', 'yearly', 'max_daily'):
        assert_agrees_with_float64(got[name], expect[name], name)


def test_maxima_are_values_the_stores_hold() -> None:
    """The spec's promise: an annual maximum is bit for bit a value in the store it was taken from."""
    dates = hours_of(1999)
    hourly = synthetic(32, dates, seed=2)
    got, rounded = fold_one_year(hourly.copy(), dates)
    for river in range(32):
        assert got['max_hourly'][river] in rounded[river], 'hourly maximum is not a value hourly.zarr holds'
        assert got['max_daily'][river] in got['daily'][river], 'daily maximum is not a value daily.zarr holds'


def test_means_cascade_by_hours_not_by_periods() -> None:
    """A yearly mean weights February like the 28 days it is, which a mean of twelve monthly means would not."""
    dates = hours_of(2019)
    hourly = synthetic(8, dates, seed=3)
    got, _ = fold_one_year(hourly.copy(), dates)
    days_in_month = np.array([pd.Period(f'2019-{m:02d}').days_in_month for m in range(1, 13)], dtype=np.float64)
    weighted = (got['monthly'].astype(np.float64) * days_in_month).sum(axis=1) / days_in_month.sum()
    unweighted = got['monthly'].astype(np.float64).mean(axis=1)
    # the yearly mean is taken from the hourly sums, so it agrees with the weighted monthly means only to the step
    # between neighbouring points of the keepbits grid, which both of them were rounded onto
    quantum = 2.0 ** -(rfs_spec.KEEP_BITS + 1)
    assert np.allclose(got['yearly'], weighted, rtol=quantum), 'the yearly mean is not weighted by month length'
    assert not np.allclose(got['yearly'], unweighted, rtol=quantum), 'a mean of monthly means is indistinguishable'


def test_float16_buffer_is_widened_not_reinterpreted() -> None:
    """A narrowed region arrives as uint16 bit patterns; folding it agrees with folding its float32 widening."""
    dates = hours_of(2003)
    hourly = synthetic(24, dates, seed=4, scale=200.0)
    narrowed = np.ascontiguousarray(hourly.astype(np.float16).view(np.uint16))
    widened = np.ascontiguousarray(narrowed.view(np.float16).astype(np.float32))

    got, _ = fold_one_year(narrowed, dates)
    expect = reference(widened, dates)
    assert np.array_equal(got['max_hourly'].view(np.uint32), expect['max_hourly'].view(np.uint32))
    for name in ('daily', 'monthly', 'yearly', 'max_daily'):
        assert_agrees_with_float64(got[name], expect[name], f'{name} from uint16')
    assert np.array_equal(narrowed.view(np.uint16), hourly.astype(np.float16).view(np.uint16)), 'uint16 was modified'


def test_float16_values_are_already_on_the_keepbits_grid() -> None:
    """Why the uint16 path skips the rounding: float16 carries fewer significand bits than keepbits keeps."""
    bits = np.arange(1, 2**16, dtype=np.uint16)
    with np.errstate(invalid='ignore'):  # the bit patterns include NaN
        widened = bits.view(np.float16).astype(np.float32)
    finite = widened[np.isfinite(widened)]
    assert np.array_equal(
        bitround(finite, rfs_spec.KEEP_BITS).view(np.uint32), finite.view(np.uint32)
    ), 'a float16 moved when rounded, so the uint16 path cannot skip rounding'


def test_float16_overflow_is_caught() -> None:
    """A region that should not have been narrowed fails on its first year rather than storing inf."""
    dates = hours_of(2005)
    hourly = synthetic(4, dates, seed=5, scale=200.0)
    hourly[2, 100] = 1e6  # above float16's 65,504 ceiling
    with np.errstate(over='ignore'):  # the overflow is the point of the test
        narrowed = np.ascontiguousarray(hourly.astype(np.float16).view(np.uint16))
    try:
        fold_one_year(narrowed, dates)
    except OverflowError as error:
        assert 'float32' in str(error)
        return
    raise AssertionError('an overflowed float16 region was folded without complaint')


def test_multi_year_accumulator_fills_left_to_right() -> None:
    """Years land in the right columns, unfolded years stay NaN, and a non-calendar year is refused."""
    years = [2018, 2019, 2020]
    accumulators = rfs_aggregate.RegionAccumulators(6, years)
    assert accumulators.daily.shape == (6, 365 + 365 + 366)
    assert accumulators.monthly.shape == (6, 36) and accumulators.yearly.shape == (6, 3)

    for year in (2018, 2020):
        dates = hours_of(year)
        accumulators.fold(dates, synthetic(6, dates, seed=year))
    assert not np.isnan(accumulators.yearly[:, [0, 2]]).any(), 'a folded year is still NaN'
    assert np.isnan(accumulators.yearly[:, 1]).all(), 'an unfolded year is not NaN'
    assert np.isnan(accumulators.daily[:, 365:730]).all(), '2019 is not the NaN block it should be'
    assert not np.isnan(accumulators.daily[:, :365]).any() and not np.isnan(accumulators.daily[:, 730:]).any()

    partial = hours_of(2019)[:-24]
    try:
        accumulators.fold(partial, synthetic(6, partial, seed=9))
    except ValueError as error:
        assert 'whole calendar year' in str(error)
        return
    raise AssertionError('a partial year was folded without complaint')


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
            print(f'ok  {name}')
