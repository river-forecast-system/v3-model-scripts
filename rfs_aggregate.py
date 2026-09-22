"""
The reductions ``2_rfs_v3_retro_router.py`` runs over a year of routed discharge while it is still in memory, and the
write queue that keeps the disk busy while the next year routes.

One year of hourly discharge for a region is the one buffer in this build that is expensive to produce and cheap to
reduce: routing it costs minutes, walking it costs a fraction of a second. Every product below hourly is a reduction
of exactly that buffer, so they are taken here rather than by reading a finished store back. ``fold_year`` makes one
pass over the year and fills every one of them at once:

    hourly[river, 8760]  ->  daily mean       [river, 365]
                         ->  monthly mean     [river, 12]
                         ->  yearly mean      [river]
                         ->  annual max of the hourly values  [river]
                         ->  annual max of the daily means    [river]

The means cascade inside the pass -- a day's hours are summed once and that sum is reused by its month and by the
year -- so every level is the mean of the *hours* under it, never a mean of means. The maxima cannot cascade: the
highest hour of a year is not recoverable from daily means, which have already flattened the peak, which is why the
hourly maximum is taken here and not by a later resampling step.

A ``RegionAccumulators`` holds those outputs for a region's whole record, every year of it, which is what lets each
published store be written exactly once per region. Sized for the largest region of the v3 hydrography (303,097
rivers) over 1940-2024, the daily array is the only large one at 18.6 GB; monthly, yearly and the two maxima come to
under 1 GB together. Writing them any other way means appending a year at a time to arrays the spec chunks as one
river over the whole time axis, where every append is a read-modify-write of the entire store.

Rounding
--------
``fold_year`` rounds each hourly value onto the spec's keepbits grid *in place* as it reads it, so the buffer handed
to the writer afterwards holds exactly the values every reduction here was taken from. That is what keeps the spec's
promise that a peak in maximums.zarr is bit for bit a value in hourly.zarr, and it makes the writer's own rounding a
no-op it can still do safely. The arithmetic matches ``river_route.router.writers.bitround`` and
``numcodecs.BitRound`` exactly; ``test_rfs_aggregate.py`` checks it against both.

A region routed at ``discharge_dtype='float16'`` arrives as uint16 bit patterns instead (numba has no float16, so
river-route carries a narrowed buffer that way). Those values hold 11 significand bits against the keepbits grid's
14, so they are already on it: the kernel widens them and skips the rounding rather than rounding what cannot move.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
from llvmlite import ir
from numba import types
from numba.extending import intrinsic, overload

import rfs_spec

__all__ = ['RegionAccumulators', 'WriteQueue', 'YearPeriods', 'FLOAT16_MAX', 'fold_year', 'keepbits_constants']

# The largest finite float16. A region routed narrowed whose discharge reaches this has overflowed to inf, and
# RegionAccumulators.fold raises rather than writing an inf into a published store.
FLOAT16_MAX = np.float32(65504.0)


@intrinsic
def _f32_bits(typingctx, x):
    """The 32 bits of a float32, as one bitcast and no instruction at all in the emitted code."""

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.IntType(32))

    return types.uint32(types.float32), codegen


@intrinsic
def _f32_from_bits(typingctx, x):
    """The float32 those 32 bits are."""

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.FloatType())

    return types.float32(types.uint32), codegen


@intrinsic
def _f16_bits_to_f32(typingctx, x):
    """The 16 bits of a float16 widened to float32, as one native conversion instruction.

    The mirror of river-route's ``_f32_to_f16_bits``, which is how a narrowed discharge buffer was filled.
    """

    def codegen(context, builder, signature, args):
        return builder.fpext(builder.bitcast(args[0], ir.HalfType()), ir.FloatType())

    return types.float32(types.uint16), codegen


def keepbits_constants(keepbits: int = rfs_spec.KEEP_BITS) -> tuple[np.uint32, np.uint32, np.uint32]:
    """
    The ``(shift, half, mask)`` the rounding below is driven by, for a given keepbits.

    They are passed into the kernel rather than baked in as globals so that changing ``rfs_spec.KEEP_BITS`` takes
    effect without clearing numba's on-disk cache, which does not track the globals a cached function closed over.
    """
    if not 0 <= keepbits < 23:
        raise ValueError(f'keepbits must be in 0..22 to round a float32, got {keepbits}')
    shift = np.uint32(23 - keepbits)
    quantum = np.uint32((1 << int(shift)) - 1)
    return shift, quantum >> np.uint32(1), np.uint32(~quantum)


def _read_hourly(row, h, shift, half, mask):
    """One hourly value as float32, rounded onto the keepbits grid and stored back when the buffer is float32."""
    raise NotImplementedError('only callable from numba')


@overload(_read_hourly, inline='always')
def _read_hourly_impl(row, h, shift, half, mask):
    """
    Compile the read for the dtype of the routed buffer.

    A float32 row is rounded and the rounded value written back, so the reductions and the file agree bit for bit.
    A uint16 row is float16 bit patterns, which carry fewer significand bits than the grid keeps, so widening them
    is the whole operation -- rounding could not move such a value and the write-back would be a store for nothing.
    """
    if isinstance(row.dtype, types.Float):

        def impl(row, h, shift, half, mask):
            bits = _f32_bits(row[h])
            bits = numba.uint32(bits + ((bits >> shift) & numba.uint32(1)) + half) & mask
            value = _f32_from_bits(bits)
            row[h] = value
            return value

        return impl
    if isinstance(row.dtype, types.Integer) and row.dtype.bitwidth == 16:

        def impl(row, h, shift, half, mask):
            return _f16_bits_to_f32(row[h])

        return impl
    raise TypeError('the hourly buffer must hold float32, or uint16 for a narrowed discharge_dtype')


@numba.njit(cache=True, nogil=True, parallel=True)
def fold_year(hourly, day_starts, month_starts, daily_out, monthly_out, yearly_out, max_hourly_out, max_daily_out,
              shift, half, mask):
    """
    Reduce one year of routed discharge into every period below it, in a single pass over the buffer.

    Each river is folded independently, so the rivers are split across threads and nothing is shared between them.
    Within a river the loop is ordered hour, day, month, year: a day's hours are summed once and that sum is added
    into its month and into the year, so a monthly mean is the mean of its hours rather than a mean of daily means,
    and no period is ever weighted by the length of the one above it.

    That cascade is what lets the sums be float32. The longest dependent chain of additions behind any value here is
    about 67 -- 24 hours into a day, 31 days into a month, 12 months into a year -- rather than the 8,784 a flat sum
    over the year would take, and error grows with the length of the chain. Measured against ``math.fsum`` over
    discharge spanning five orders of magnitude, the worst relative error is 3.2e-07 daily and 2.2e-07 yearly,
    against the 6.1e-05 step between neighbouring points of the keepbits grid every value is then rounded onto: a
    margin of about 190x. float64 sums measured 5.1e-08, which after that rounding is the same number.

    Summing and then dividing, rather than dividing each hour and adding, is deliberate. Dividing first does not
    shorten the chain, it just rounds once per hour before the same summation and then has to weight each period's
    contribution to the one above it; measured the same way it is worse, 1.1e-06 on the yearly mean.

    Args:
        hourly: (river, hour) routed discharge, float32, or uint16 holding float16 bit patterns. A float32 buffer is
            rounded onto the keepbits grid in place, so pass it to the writer after this returns, never before.
        day_starts: (n_days + 1,) offsets into the hour axis where each day begins, the last being its end
        month_starts: (n_months + 1,) offsets into ``day_starts`` where each month begins, the last being its end
        daily_out: (river, n_days) float32 destination for the daily means
        monthly_out: (river, n_months) float32 destination for the monthly means
        yearly_out: (river,) float32 destination for the mean of the whole span
        max_hourly_out: (river,) float32 destination for the largest hourly value
        max_daily_out: (river,) float32 destination for the largest daily mean
        shift: bits of mantissa the rounding discards, from ``keepbits_constants``
        half: the tie-breaking increment, from ``keepbits_constants``
        mask: the bits the rounding keeps, from ``keepbits_constants``
    """
    n_rivers = hourly.shape[0]
    n_months = month_starts.shape[0] - 1
    span_hours = np.float32(day_starts[month_starts[n_months]] - day_starts[month_starts[0]])
    for i in numba.prange(n_rivers):
        row = hourly[i]
        span_sum = np.float32(0.0)
        peak_hour = np.float32(-np.inf)
        peak_day = np.float32(-np.inf)
        for m in range(n_months):
            d_lo, d_hi = month_starts[m], month_starts[m + 1]
            month_sum = np.float32(0.0)
            for d in range(d_lo, d_hi):
                h_lo, h_hi = day_starts[d], day_starts[d + 1]
                day_sum = np.float32(0.0)
                for h in range(h_lo, h_hi):
                    value = _read_hourly(row, h, shift, half, mask)
                    day_sum += value
                    if value > peak_hour:
                        peak_hour = value
                # the daily mean is rounded before it is compared, so the annual maximum of the daily series is a
                # value daily.zarr holds rather than one a later rounding would move off the peak
                day_mean = _round_f32(np.float32(day_sum / (h_hi - h_lo)), shift, half, mask)
                daily_out[i, d] = day_mean
                if day_mean > peak_day:
                    peak_day = day_mean
                month_sum += day_sum
            month_hours = day_starts[d_hi] - day_starts[d_lo]
            monthly_out[i, m] = _round_f32(np.float32(month_sum / month_hours), shift, half, mask)
            span_sum += month_sum
        yearly_out[i] = _round_f32(np.float32(span_sum / span_hours), shift, half, mask)
        max_hourly_out[i] = peak_hour
        max_daily_out[i] = peak_day


@numba.njit(cache=True, nogil=True, inline='always')
def _round_f32(value, shift, half, mask):
    """A float32 with all but the leading keepbits mantissa bits zeroed, rounded to nearest even."""
    bits = _f32_bits(value)
    return _f32_from_bits(numba.uint32(bits + ((bits >> shift) & numba.uint32(1)) + half) & mask)


@dataclass(frozen=True)
class YearPeriods:
    """Where the days and months of one routed year fall, in the year and in the region's whole record."""

    day_starts: np.ndarray  # (n_days + 1,) offsets into the year's hour axis
    month_starts: np.ndarray  # (n_months + 1,) offsets into day_starts
    day_offset: int  # index of this year's first day in the record's daily axis
    month_offset: int  # index of its first month in the monthly axis
    year_index: int  # index of the year itself in the yearly axis

    @property
    def n_days(self) -> int:
        return self.day_starts.shape[0] - 1

    @property
    def n_months(self) -> int:
        return self.month_starts.shape[0] - 1


def _boundaries(labels: pd.PeriodIndex) -> np.ndarray:
    """Offsets where each run of equal labels begins, with the length of the index appended as the final end."""
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    return np.r_[starts, labels.shape[0]].astype(np.int64)


class RegionAccumulators:
    """
    Every product below hourly, for one region over its whole record, held in memory until the region is finished.

    A region process routes its years in order and folds each one in as it lands, so the arrays fill left to right
    and are complete once the last year is routed. They are then written out in one call per store, which is the
    point of holding them: every chunk of a published store is written exactly once, and nothing is ever read back
    to be appended to.

    Only complete calendar periods are on these axes, which for whole calendar years of hourly routing means every
    period is complete by construction. ``fold`` raises on a year that is not a whole calendar year rather than
    recording a mean over the part of a period the record has.

    Slots for years that have not been folded in hold NaN, the same fill value the published stores use, so a run
    that dies part way leaves a store that reads as missing rather than as zero discharge.
    """

    def __init__(self, n_rivers: int, years: list[int] | range) -> None:
        years = [int(y) for y in years]
        if sorted(years) != years or len(set(years)) != len(years):
            raise ValueError('years must be given in ascending order with no repeats')
        if years != list(range(years[0], years[-1] + 1)):
            raise ValueError(f'the record has a gap: {years[0]}..{years[-1]} is not every year in between')
        self.years = years
        self.n_rivers = n_rivers
        start, end = pd.Timestamp(f'{years[0]}-01-01'), pd.Timestamp(f'{years[-1] + 1}-01-01')
        self.daily_times = pd.date_range(start, end, freq='D', inclusive='left')
        self.monthly_times = pd.date_range(start, end, freq='MS', inclusive='left')
        self.yearly_times = pd.date_range(start, end, freq='YS', inclusive='left')
        day_starts = np.searchsorted(self.daily_times, self.yearly_times)
        month_starts = np.searchsorted(self.monthly_times, self.yearly_times)
        self._day_of_year = {y: int(i) for y, i in zip(years, day_starts, strict=True)}
        self._month_of_year = {y: int(i) for y, i in zip(years, month_starts, strict=True)}
        shape = (n_rivers, len(self.daily_times))
        self.daily = np.full(shape, np.nan, dtype=np.float32)
        self.monthly = np.full((n_rivers, len(self.monthly_times)), np.nan, dtype=np.float32)
        self.yearly = np.full((n_rivers, len(years)), np.nan, dtype=np.float32)
        self.max_hourly = np.full((n_rivers, len(years)), np.nan, dtype=np.float32)
        self.max_daily = np.full((n_rivers, len(years)), np.nan, dtype=np.float32)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.daily, self.monthly, self.yearly, self.max_hourly, self.max_daily))

    def periods(self, dates: pd.DatetimeIndex) -> YearPeriods:
        """Where one routed year's days and months fall. Raises unless it is exactly one calendar year of hours."""
        dates = pd.DatetimeIndex(dates)
        year = dates[0].year
        if year not in self._day_of_year:
            raise ValueError(f'{year} is outside the record this accumulator covers, {self.years[0]}..{self.years[-1]}')
        expected = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left')
        if not dates.equals(expected):
            raise ValueError(
                f'{year} is not one whole calendar year of hourly steps: got {len(dates)} steps '
                f'{dates[0]}..{dates[-1]}, expected {len(expected)} steps {expected[0]}..{expected[-1]}'
            )
        day_starts = _boundaries(dates.to_period('D'))
        return YearPeriods(
            day_starts=day_starts,
            # the months are indexed in days, not hours, so they are labelled from each day's first timestamp
            month_starts=_boundaries(dates[day_starts[:-1]].to_period('M')),
            day_offset=self._day_of_year[year],
            month_offset=self._month_of_year[year],
            year_index=self.years.index(year),
        )

    def fold(self, dates: pd.DatetimeIndex, hourly: np.ndarray) -> YearPeriods:
        """
        Reduce one routed year into every accumulator, and round the hourly buffer onto the keepbits grid in place.

        The buffer is modified, so it is handed to the writer after this returns and never before. Returns the
        periods it folded, whose ``year_index`` is the column a caller can read the year's maxima back out of.
        """
        if hourly.shape[0] != self.n_rivers:
            raise ValueError(f'{hourly.shape[0]} rivers of discharge for a {self.n_rivers} river accumulator')
        if hourly.shape[1] != len(dates):
            raise ValueError(f'{hourly.shape[1]} hourly steps against {len(dates)} dates')
        period = self.periods(dates)
        d0, d1 = period.day_offset, period.day_offset + period.n_days
        m0, m1 = period.month_offset, period.month_offset + period.n_months
        y = period.year_index
        shift, half, mask = keepbits_constants()
        fold_year(
            hourly, period.day_starts, period.month_starts,
            self.daily[:, d0:d1], self.monthly[:, m0:m1],
            self.yearly[:, y], self.max_hourly[:, y], self.max_daily[:, y],
            shift, half, mask,
        )
        if hourly.dtype != np.float32:
            # A narrowed region whose discharge left float16's range stored inf, which would otherwise be written
            # into a published store and fitted against by return-periods.zarr. Caught on the first year a region
            # routes, so a misclassified region costs one year rather than a whole record.
            peak = float(np.nanmax(self.max_hourly[:, y]))
            if not np.isfinite(peak):
                raise OverflowError(
                    f'{dates[0].year} overflowed float16: discharge above {FLOAT16_MAX:,.0f} m3 s-1 cannot be '
                    f'stored narrowed. Route this region at discharge_dtype="float32".'
                )
        return period


class WriteQueue:
    """
    A bounded pool of writer threads, so a year's discharge goes to disk while the next year is routing.

    Routing a year is minutes of compute and writing it is tens of seconds of disk; done in sequence the CPU waits
    for the disk once a year, and the disk is idle the rest of the time. ``submit`` hands the write to a thread and
    returns, so the two overlap. zarr's writers release the GIL while blosc compresses, which is what makes threads
    rather than processes the right tool -- and means a task can share the routed buffer instead of pickling it.

    ``depth`` is the back pressure, and it is a memory budget rather than a tuning knob: a year of hourly discharge
    for the largest v3 region is 10.6 GB float32, so a queue that accepts years faster than the disk drains them
    would grow without bound. ``submit`` blocks once ``depth`` writes are in flight, which holds the process to
    ``depth + 1`` year buffers. Overlapping therefore costs one whole year buffer, which is nothing against a
    region and a great deal against the 172.8 GB a year of every river in the world takes: a caller with no room
    for the second buffer passes no queue at all and writes inline, trading the overlap for the memory.

    An exception in a writer is re-raised by the next ``submit`` or by ``close``, so a failed write stops the run
    instead of leaving a store with a hole in it. Use it as a context manager; ``close`` waits for the queue to
    drain, which is what makes the stores complete when the ``with`` block ends.
    """

    def __init__(self, workers: int = 2, depth: int = 2) -> None:
        if workers < 1 or depth < 1:
            raise ValueError(f'workers and depth must both be >= 1, got {workers} and {depth}')
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix='rfs-write')
        self._slots = threading.BoundedSemaphore(depth)
        self._errors: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False

    def __enter__(self) -> WriteQueue:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _raise_first_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise error

    @staticmethod
    def inline(fn, /, *args, **kwargs) -> None:
        """Run a write on the calling thread. What ``submit`` degrades to when there is no room to overlap."""
        fn(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs) -> None:
        """Queue one write, blocking while ``depth`` of them are already in flight."""
        if self._closed:
            raise RuntimeError('this WriteQueue is closed')
        self._raise_first_error()
        self._slots.acquire()

        def run() -> None:
            try:
                fn(*args, **kwargs)
            except BaseException as error:  # noqa: BLE001 - recorded and re-raised on the submitting thread
                self._errors.put(error)
            finally:
                self._slots.release()

        self._pool.submit(run)

    def close(self) -> None:
        """Wait for every queued write to finish, then re-raise the first one that failed."""
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True)
        self._raise_first_error()
