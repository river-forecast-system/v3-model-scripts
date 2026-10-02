"""
Concatenate the hourly discharge 1_route_regions.py writes, one zarr per region per calendar year, into the published
whole-record stores, and upload them to S3 while they are written:

    <out-dir>/hourly.zarr      Q: every hour of the record
    <out-dir>/daily.zarr       Q: the daily means
    <out-dir>/monthly.zarr     Q: the monthly means, and Q_timesteps
    <out-dir>/yearly.zarr      Q: the yearly means, and Q_timesteps
    <out-dir>/maximums.zarr    hourly and daily: each year's largest hourly value and largest daily mean

laid out as rfs_spec.py says, with the values of rfs_spec.reduce_year. It is built so that the ~15 TB of hourly record
is read once and written once, and every reduction is made in the same pass.

hourly and daily are written as the record streams through. An hourly shard is 250 rivers over every year, a daily
shard 1,000, and an input chunk is 500 rivers of one year, so no shard can be written until every year of its rivers
is read. The rivers are cut into segments of whole daily shards, and a process streams through its segment in
riverIndex order: it reads the next input chunk of every year into a buffer, reduces and writes every hourly shard the
buffer now completes, and carries the rest, fewer than 250 rivers, on to the next chunk. The daily means wait in a
second buffer until they fill a daily shard. The hourly buffer holds at most 749 rivers of the whole record, 2.2 GB,
and nothing is written twice or read back. The only input read twice is the chunk each cut between segments falls in.
Segments shrink as the work runs out, each a 1 / (2 * --processes) share of what is left and at least
MIN_SEGMENT_UNITS daily shards, so the processes finish together.

Whole shards of one-river chunks are encoded here, a river per blosc call on --threads threads, and written straight to
their files, byte for byte what zarr writes. zarr costs about 350 us per chunk whatever its size, which is lost in the
compression of an hourly chunk but not of a monthly one. Each process first checks on a small shard that shard_bytes
reproduces zarr's bytes for the array's codecs, and writes through zarr if it does not. The last shard, which holds
fewer rivers than the others, always goes through zarr.

monthly, yearly and maximums are hundreds or thousands of times smaller than hourly, and their shards are tens of
thousands of rivers, more than a segment's cut can be put between. So each segment writes their values into .npy files
in --work-dir, 25 GB together for 85 years, and they are written from those once every segment is done, along with
Q_timesteps, which is every river at one step and cannot be written until every segment is either.

With --s3-url, every file of the stores is uploaded to the same key under that prefix. The main process uploads each
segment's shards as it reports them written, on --upload-threads threads, from files still in the page cache, so the
build never waits on the network. The metadata goes first, so a bad bucket or credentials fail before anything is
built. Each uploaded file is logged in --work-dir, and a rerun uploads only what the log lacks.

A rerun resumes. Each segment records in --work-dir the rivers it has written, after their shards and .npy rows, and
continues from there. Nothing is ever deleted: to build the stores again, delete them and --work-dir first.
"""

import argparse
import json
import os
import re
import shutil
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path

import google_crc32c
import numcodecs
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from zarr.core.buffer.cpu import NDBuffer

import rfs_spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

EMIT = rfs_spec.LAYOUT['hourly'][1]  # rivers reduced and written at once: one hourly shard
UNIT = rfs_spec.LAYOUT['daily'][1]  # rivers a segment is cut on: one daily shard, a whole number of hourly ones
# the smallest segment, 10 daily shards or 10,000 rivers: its cuts reread at most two of river-route's 500 river chunks
MIN_SEGMENT_UNITS = 10
POLL_SECONDS = 5
MULTIPART_BYTES = 64 * 2**20  # files larger than this are uploaded in parts of this size
# 'hourly' for the record and each reduction by rfs_spec.reduce_year's name: the store and variable it is written to.
# STREAMED are written as each segment streams, KEPT are kept in --work-dir and written once every segment is done
STREAMED = {'hourly': ('hourly', 'Q'), 'daily': ('daily', 'Q')}
KEPT = {'monthly': ('monthly', 'Q'), 'yearly': ('yearly', 'Q'), 'max_hourly': ('maximums', 'hourly'),
        'max_daily': ('maximums', 'daily')}
SHUFFLES = {'noshuffle': numcodecs.Blosc.NOSHUFFLE, 'shuffle': numcodecs.Blosc.SHUFFLE,
            'bitshuffle': numcodecs.Blosc.BITSHUFFLE}
TIMESTEPS = ('monthly', 'yearly')  # the stores with Q_timesteps, written from the same kept values as their Q
# river-route names the discharge of runoff file era5_194001_194012.zarr discharge_era5_194001_194012.zarr
DISCHARGE = re.compile(r'discharge_(?P<stem>.*_(?P<year>\d{4})01_(?P=year)12)\.zarr')


def read_dates(time_var: zarr.Array) -> pd.DatetimeIndex:
    """The dates of a time variable as river-route's writers encode them, with units '<unit> since <date>'."""
    unit, origin = time_var.attrs['units'].split(' since ')
    return pd.Timestamp(origin) + pd.to_timedelta(time_var[:], unit=unit)


def check_input(path: Path, river_ids: np.ndarray, year: int) -> tuple[int, int]:
    """Check one region's year of discharge, and return how many hours after the year it starts and its rivers per
    chunk."""
    group = zarr.open_group(str(path), mode='r')
    q = group['Q']
    if not np.array_equal(group['river_id'][:], river_ids):
        raise ValueError(f'{path} does not hold its region in riverIndex order')
    dates = read_dates(group['time'])
    hours = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left')
    if not np.array_equal(dates, hours[hours.size - dates.size:]):
        raise ValueError(f'{path}: {dates[0]}..{dates[-1]} is not every hour of {year} from its first')
    if q.dtype != np.float32 or q.shape != (river_ids.size, dates.size) or q.chunks[1] != dates.size:
        raise ValueError(f'{path}: Q is {q.dtype} {q.shape} in chunks of {q.chunks}, not float32 chunks of whole years')
    return hours.size - dates.size, q.chunks[0]


def plan_segments(n_rivers: int, processes: int) -> list[tuple[int, int]]:
    """Cut the rivers into segments of whole daily shards, largest first: each is 1 / (2 * processes) of the shards left
    and at least MIN_SEGMENT_UNITS, so the last to start are small and the processes finish together."""
    units = -(-n_rivers // UNIT)
    cuts = [0]
    while cuts[-1] < units:
        left = units - cuts[-1]
        cuts.append(cuts[-1] + min(left, max(MIN_SEGMENT_UNITS, -(-left // (2 * processes)))))
    return [(a * UNIT, min(b * UNIT, n_rivers)) for a, b in zip(cuts, cuts[1:], strict=False)]


def progress(work_dir: Path, segment: int, first: int) -> int:
    """The first river of the segment not yet written."""
    path = work_dir / 'progress' / str(segment)
    return int(path.read_text()) if path.exists() else first


def start_worker(threads: int) -> None:
    """Size zarr's thread pool, which each process creates once, at first use, and keep blosc to one thread per call:
    the calls are already spread over threads."""
    zarr.config.set({'async.concurrency': threads, 'threading.max_workers': threads})
    numcodecs.blosc.use_threads = False


def shard_bytes(chunks: list[bytes]) -> list[bytes]:
    """
    A shard of encoded chunks as zarr's sharding codec lays it out with its default index at the end: the chunks in
    order, which is zarr's morton order for a column of one-river chunks, then each chunk's (offset, nbytes) as
    little-endian uint64 and the crc32c of those.
    """
    lengths = np.array([len(chunk) for chunk in chunks], dtype='<u8')
    index = np.column_stack((np.cumsum(lengths) - lengths, lengths)).astype('<u8').tobytes()
    return [*chunks, index, google_crc32c.value(index).to_bytes(4, 'little')]


def shard_codec(array: zarr.Array) -> numcodecs.Blosc | None:
    """
    The blosc codec to encode the array's whole shards with, or None if they must be written through zarr: when a
    shard is not one-river chunks over the whole time axis, or when shard_bytes does not reproduce what zarr writes
    for the array's compressor, checked on a shard of random values.
    """
    if array.chunks != (1, array.shape[1]) or array.shards[1] != array.shape[1] or len(array.compressors) != 1:
        return None
    (codec,) = array.compressors
    if not isinstance(codec, zarr.codecs.BloscCodec):
        return None
    blosc = numcodecs.Blosc(cname=codec.cname, clevel=codec.clevel, shuffle=SHUFFLES[codec.shuffle],
                            blocksize=codec.blocksize, typesize=codec.typesize)
    probe, values = {}, np.random.default_rng(0).lognormal(3, 2, (array.shards[0], 100)).astype(array.dtype)
    zarr.create_array(store=zarr.storage.MemoryStore(probe), shape=values.shape, chunks=(1, 100), shards=values.shape,
                      dtype=array.dtype, fill_value=array.fill_value, compressors=array.compressors,
                      config={'write_empty_chunks': True})[:] = values
    return blosc if b''.join(shard_bytes([blosc.encode(r) for r in values])) == probe['c/0/0'].to_bytes() else None


def write_rivers(array: zarr.Array, codec: numcodecs.Blosc | None, path: Path, values: np.ndarray, row: int,
                 pool: ThreadPoolExecutor, threads: int) -> None:
    """Write rivers [row, row + len(values)) of the array stored at path, row on a shard boundary: with a codec, whole
    shards encoded here and written straight to their files, and a last shard of fewer rivers through zarr; without
    one, everything through zarr."""
    width = array.shards[0]
    whole = len(values) // width * width if codec else 0
    if whole:
        bounds = np.linspace(0, whole, threads + 1).astype(int)
        parts = pool.map(lambda a, b: [codec.encode(river) for river in values[a:b]], bounds[:-1], bounds[1:])
        chunks = [chunk for part in parts for chunk in part]
        for k in range(0, whole, width):
            file = path / array.metadata.encode_chunk_key(((row + k) // width, 0))
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, 'wb') as handle:
                handle.writelines(shard_bytes(chunks[k:k + width]))
    if whole < len(values):
        array[row + whole:row + len(values)] = values[whole:]


def build_segment(job: dict) -> None:
    """Stream one segment of rivers, in riverIndex order, from the input chunks into whole shards of hourly and daily,
    and the kept values of the other stores."""
    segment, (first, end), columns = job['segment'], job['rows'], job['columns']
    start = progress(job['work_dir'], segment, first)
    if start >= end:
        return
    n_years, n_days = len(columns), columns[-1][4]
    out, work, threads = job['out_dir'], job['work_dir'], job['threads']
    paths = {name: out / f'{store}.zarr' / variable for name, (store, variable) in STREAMED.items()}
    targets = {name: zarr.open_array(str(path), mode='r+') for name, path in paths.items()}
    codecs = {name: shard_codec(array) for name, array in targets.items()}
    kept = {name: np.load(work / f'{name}.npy', mmap_mode='r+') for name in KEPT}
    widths = {'daily': n_days, 'monthly': 12 * n_years, 'yearly': n_years, 'max_hourly': n_years,
              'max_daily': n_years}

    # rows are rivers from `row` on, columns every hour of the record. Only the hours before the first year's input
    # starts are never read into it, so they are the only ones set here
    buffer = np.empty((EMIT - 1 + job['max_block'], targets['hourly'].shape[1]), np.float32)
    buffer[:, :columns[0][1]] = np.nan
    # the daily means of rivers from `daily_row` on, reduced but not yet written: fewer than a daily shard between emits
    daily = np.empty((UNIT - 1 + buffer.shape[0], n_days), np.float32)
    daily_row, pending = start, 0
    sources = {}  # the Q array of every year of the region being read

    def read(region: str, l0: int, l1: int, at: int, pool: ThreadPoolExecutor) -> None:
        """Decode rivers [l0, l1) of the region, every year, straight into buffer rows [at, at + l1 - l0)."""
        if region not in sources:
            sources.clear()
            sources[region] = [zarr.open_array(str(path / 'Q'), mode='r') for path in job['inputs'][region]]

        def one(i: int) -> None:
            _, h0, h1, _, _ = columns[i]
            out_view = NDBuffer.from_numpy_array(buffer[at:at + l1 - l0, h0:h1])
            sources[region][i].get_basic_selection((slice(l0, l1), slice(None)), out=out_view)

        list(pool.map(one, range(n_years)))

    def emit(n: int, row: int, pool: ThreadPoolExecutor) -> None:
        """Reduce buffer rows [0, n), rounding them onto the keepbits grid, write them as hourly rivers [row, row + n),
        keep their reductions, and write every daily shard now complete, or every daily row at the segment's end."""
        nonlocal daily_row, pending
        rows = buffer[:n]
        reduced = {name: np.empty((n, width), np.float32) for name, width in widths.items()}

        def one(i: int) -> None:
            year, h0, h1, d0, d1 = columns[i]
            values = rfs_spec.reduce_year(rows[:, h0:h1], year)
            reduced['daily'][:, d0:d1] = values['daily']
            reduced['monthly'][:, 12 * i:12 * i + 12] = values['monthly']
            for name in ('yearly', 'max_hourly', 'max_daily'):
                reduced[name][:, i] = values[name]

        list(pool.map(one, range(n_years)))
        write_rivers(targets['hourly'], codecs['hourly'], paths['hourly'], rows, row, pool, threads)
        for name, values in kept.items():
            values[row:row + n] = reduced[name]
        daily[pending:pending + n] = reduced['daily']
        pending += n
        k = pending if row + n == end else pending // UNIT * UNIT
        if k:
            write_rivers(targets['daily'], codecs['daily'], paths['daily'], daily[:k], daily_row, pool, threads)
            daily[:pending - k] = daily[k:pending]
            daily_row, pending = daily_row + k, pending - k

    record = job['work_dir'] / 'progress' / str(segment)
    row, filled = start, 0  # buffer row 0 is river `row`, and rows [0, filled) are read but not yet written
    with ThreadPoolExecutor(threads) as pool:
        for region, l0, l1, g0 in job['blocks']:
            lo, hi = max(g0, start), min(g0 + l1 - l0, end)
            if hi <= lo:
                continue
            read(region, l0 + lo - g0, l0 + hi - g0, filled, pool)
            filled += hi - lo
            n = filled if row + filled == end else filled // EMIT * EMIT
            if not n:
                continue
            written = daily_row
            emit(n, row, pool)
            buffer[:filled - n] = buffer[n:filled]  # fewer than EMIT rows from at least EMIT on: they never overlap
            row, filled = row + n, filled - n
            if daily_row > written:  # every store has its rivers up to daily_row, which a rerun resumes from
                partial = record.with_suffix('.partial')
                partial.write_text(str(daily_row))
                os.replace(partial, record)
    if row != end or filled or pending:
        raise RuntimeError(f'segment {segment} ended at river {row} with {filled + pending} unwritten, not at {end}')


def write_kept(out_dir: Path, work_dir: Path, threads: int) -> None:
    """Write the monthly, yearly and maximums arrays from the values kept in work_dir, whole shards at a time, and
    then Q_timesteps."""
    with ThreadPoolExecutor(threads) as pool:
        for name, (store, variable) in KEPT.items():
            path = out_dir / f'{store}.zarr' / variable
            array = zarr.open_array(str(path), mode='r+')
            codec, values = shard_codec(array), np.load(work_dir / f'{name}.npy', mmap_mode='r')
            step = array.shards[0] * threads  # a shard a thread, or as many as zarr writes at once
            for r0 in range(0, values.shape[0], step):
                write_rivers(array, codec, path, np.asarray(values[r0:r0 + step]), r0, pool, threads)
    write_timesteps(out_dir, work_dir, threads)


def write_timesteps(out_dir: Path, work_dir: Path, threads: int) -> None:
    """Fill Q_timesteps from the monthly and yearly values kept in work_dir, TIMESTEPS_CHUNK[0] rivers at a time."""
    width = rfs_spec.TIMESTEPS_CHUNK[0]
    n_rivers = np.load(work_dir / f'{TIMESTEPS[0]}.npy', mmap_mode='r').shape[0]

    def put(name: str, r0: int) -> None:
        dst = zarr.open_array(str(out_dir / f'{name}.zarr' / 'Q_timesteps'), mode='r+')
        dst[r0:r0 + width] = np.load(work_dir / f'{name}.npy', mmap_mode='r')[r0:r0 + width]

    blocks = [(name, r0) for name in TIMESTEPS for r0 in range(0, n_rivers, width)]
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(put, *zip(*blocks, strict=True)))


def shard_files(arrays: dict, g0: int, g1: int) -> list[str]:
    """The files, relative to the out dir, of every sharded array's shards that hold rivers [g0, g1)."""
    files = []
    for (name, variable), array in arrays.items():
        rows, cols = array.shards
        for i in range(g0 // rows, -(-g1 // rows)):
            files += [f'{name}.zarr/{variable}/{array.metadata.encode_chunk_key((i, j))}'
                      for j in range(-(-array.shape[1] // cols))]
    return files


def store_files(out_dir: Path, data: bool) -> list[str]:
    """The files of every store relative to out_dir: its data chunks, or everything else, which create_store writes.
    Chunks are under <variable>/c/, zarr v3's default chunk key encoding."""
    chunk_dirs = tuple(f'{name}.zarr/{variable}/c/' for name, (_, variables, _, timesteps) in rfs_spec.STORES.items()
                       for variable in (*variables, *(('Q_timesteps',) if timesteps else ())))
    files = []
    for name in rfs_spec.STORES:
        for directory, _, names in os.walk(out_dir / f'{name}.zarr'):
            rel = Path(directory).relative_to(out_dir).as_posix() + '/'
            if rel.startswith(chunk_dirs) == data:
                files += [rel + n for n in names]
    return files


class Uploader:
    """Upload files under a root directory to the same keys under an S3 prefix, each once. What is uploaded is logged,
    so a rerun skips it."""

    def __init__(self, url: str, endpoint: str | None, root: Path, log: Path, threads: int, storage_class: str | None):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        if not url.startswith('s3://'):
            raise SystemExit(f'--s3-url {url} is not s3://bucket/prefix')
        self.bucket, _, prefix = url.removeprefix('s3://').partition('/')
        self.prefix = prefix.strip('/') + '/' if prefix.strip('/') else ''
        self.root, self.extra = root, {'StorageClass': storage_class} if storage_class else {}
        config = Config(max_pool_connections=4 * threads, retries={'max_attempts': 10, 'mode': 'adaptive'})
        self.client = boto3.client('s3', endpoint_url=endpoint, config=config)
        self.transfer = TransferConfig(multipart_threshold=MULTIPART_BYTES, multipart_chunksize=MULTIPART_BYTES,
                                       max_concurrency=4)
        self.pool = ThreadPoolExecutor(threads)
        self.done = set(log.read_text().split()) if log.exists() else set()
        self.log, self.lock = log, threading.Lock()
        self.queued, self.futures, self.failed, self.sent, self.bytes = set(), [], [], 0, 0

    def submit(self, files: list[str]) -> None:
        for rel in files:
            if rel not in self.done and rel not in self.queued:
                self.queued.add(rel)
                self.futures.append(self.pool.submit(self._put, rel))

    def _put(self, rel: str) -> None:
        path, key = self.root / rel, self.prefix + rel
        try:
            size = path.stat().st_size
            if size > MULTIPART_BYTES:
                self.client.upload_file(str(path), self.bucket, key, ExtraArgs=self.extra, Config=self.transfer)
            else:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=path.read_bytes(), **self.extra)
        except Exception as error:
            with self.lock:
                self.failed.append(rel)
            tqdm.write(f'upload of {rel} failed: {error!r}')
            return
        with self.lock, open(self.log, 'a') as log:
            log.write(rel + '\n')
            self.sent, self.bytes = self.sent + 1, self.bytes + size

    def pending(self) -> int:
        return sum(not f.done() for f in self.futures)

    def status(self) -> str:
        return f'uploaded {self.sent:,} of {len(self.queued):,} files, {self.bytes / 1e9:,.1f} GB'

    def close(self) -> None:
        self.pool.shutdown(wait=True)


if __name__ == '__main__':
    began = time.time()
    cores = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hydrography', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'hydrography')
    parser.add_argument('--routing', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'routing',
                        help='where 1_prepare_inputs/ wrote each region\'s routing files')
    parser.add_argument('--discharge-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'discharge',
                        help='where 1_route_regions.py wrote the hourly discharge')
    parser.add_argument('--out-dir', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'retrospective')
    parser.add_argument('--work-dir', type=Path, help='progress, the upload log, and the monthly, yearly and maximum '
                                                      'values those stores are written from, 25 GB for 85 years. '
                                                      'Default <out-dir>/.work')
    parser.add_argument('--processes', type=int, default=max(1, cores // 4), help='segments built at once')
    parser.add_argument('--threads', type=int, default=4, help='threads each process decodes, reduces and encodes on')
    parser.add_argument('--s3-url', help='s3://bucket/prefix to upload the stores to as they are written')
    parser.add_argument('--s3-endpoint-url', help='an S3 compatible endpoint to upload to instead of AWS')
    parser.add_argument('--storage-class', help='S3 storage class of every upload, e.g. INTELLIGENT_TIERING. '
                                                'Default the bucket\'s')
    parser.add_argument('--upload-threads', type=int, default=16, help='files uploaded at once')
    args = parser.parse_args()
    work = args.work_dir or args.out_dir / '.work'
    if args.processes < 1 or args.threads < 1 or args.processes * args.threads > cores:
        raise SystemExit(f'--processes {args.processes} x --threads {args.threads} must be at least 1 and at most '
                         f'the {cores} cores of this machine')
    zarr.config.set({'async.concurrency': cores, 'threading.max_workers': cores})

    regions = sorted(d.name.split('=')[1] for d in args.routing.glob('region=*'))
    regions = [r for r in regions if r != 'global']  # region=global is every river as one network
    river_ids, offsets = rfs_spec.global_layout(args.hydrography, args.routing, regions)
    regions.sort(key=lambda r: offsets[r][0])  # riverIndex order, the order the segments stream in

    inputs = {r: {} for r in regions}  # region: {year: its discharge zarr}, for every year it has finished routing
    for region in regions:
        for path in sorted((args.discharge_root / f'region={region}').glob('discharge_*.zarr')):
            if not (match := DISCHARGE.fullmatch(path.name)):
                raise SystemExit(f'{path} is not the discharge of one calendar year of runoff')
            if (path.parent / f'channel_state_{match["stem"]}.parquet').exists():
                inputs[region][int(match['year'])] = path
    found = sorted({year for years in inputs.values() for year in years})
    if not found:
        raise SystemExit(f'no finished years of discharge in {args.discharge_root}')
    years = list(range(found[0], found[-1] + 1))
    if missing := [f'{r} {y}' for r in regions for y in years if y not in inputs[r]]:
        raise SystemExit(f'{len(missing)} region years of {years[0]}..{years[-1]} are not routed, the first '
                         f'{missing[0]}. Every one is needed for the whole record')

    pairs = [(r, y) for r in regions for y in years]
    with ThreadPoolExecutor(cores) as pool:
        checked = dict(zip(pairs, pool.map(
            lambda p: check_input(inputs[p[0]][p[1]], river_ids[slice(*offsets[p[0]])], p[1]), pairs), strict=True))
    late = {y: {checked[r, y][0] for r in regions} for y in years}
    if any(len(late[y]) > 1 for y in years) or any(late[y] != {0} for y in years[1:]):
        raise SystemExit('only the first year may start after its first hour, and at the same hour in every region')
    chunk_rows = {r: {checked[r, y][1] for y in years} for r in regions}
    if any(len(rows) > 1 for rows in chunk_rows.values()):
        raise SystemExit('a region is chunked differently in different years')
    chunk_rows = {r: rows.pop() for r, rows in chunk_rows.items()}

    times = rfs_spec.record_times(years[0], years[-1])
    columns = []  # (year, its first hour read, its end hour, its first day, its end day), as columns of the stores
    for year in years:
        h0, h1 = times['hourly'].searchsorted([pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year + 1}-01-01')])
        d0, d1 = times['daily'].searchsorted([pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year + 1}-01-01')])
        columns.append((year, int(h0) + late[year].pop(), int(h1), int(d0), int(d1)))
    blocks = []  # (region, its first row, its end row, the riverIndex of the first): one input chunk, riverIndex order
    for region in regions:
        start, end = offsets[region]
        blocks += [(region, l0, min(l0 + chunk_rows[region], end - start), start + l0)
                   for l0 in range(0, end - start, chunk_rows[region])]

    stores = {name: args.out_dir / f'{name}.zarr' for name in rfs_spec.STORES}
    layout_file, layout = work / 'layout.json', {'rivers': int(river_ids.size), 'years': [years[0], years[-1]],
                                                 'kept': list(KEPT), 'unit': UNIT}
    if layout_file.exists():
        saved = json.loads(layout_file.read_text())
        if {key: saved.get(key) for key in layout} != layout:
            raise SystemExit(f'{work} is for {saved["rivers"]:,} rivers over {saved["years"]}: delete it and the '
                             f'stores in {args.out_dir} to build them again')
        segments = [tuple(s) for s in saved['segments']]
    else:
        if existing := [str(p) for p in stores.values() if p.exists()]:
            raise SystemExit(f'{existing[0]} exists but {layout_file} does not, so it cannot be resumed: '
                             f'delete the stores in {args.out_dir} to build them again')
        (work / 'progress').mkdir(parents=True, exist_ok=True)
        segments = plan_segments(river_ids.size, args.processes)
        for name, path in stores.items():
            rfs_spec.create_store(path, name, times[name], river_ids)
        for name, (store, _) in KEPT.items():
            np.lib.format.open_memmap(work / f'{name}.npy', mode='w+', dtype=np.float32,
                                      shape=(river_ids.size, len(times[store])))
        layout_file.write_text(json.dumps({**layout, 'segments': segments}))  # last, so it marks all of the above

    raw = river_ids.size * len(times['hourly']) * 4
    per_process = (EMIT - 1 + max(chunk_rows.values())) * len(times['hourly']) * 4
    sample = sum(f.stat().st_size for r in regions for f in (inputs[r][years[-1]] / 'Q').rglob('*') if f.is_file())
    free = shutil.disk_usage(args.out_dir).free
    print(f'{river_ids.size:,} rivers in {len(regions)} regions, {years[0]}..{years[-1]}: {raw / 1e12:.1f} TB of '
          f'hourly record before compression, about {sample * len(years) / 1e12:.1f} TB of input after it. '
          f'{free / 1e12:.1f} TB free in {args.out_dir}', flush=True)
    # the stores come out about the size of the input: hourly at the input's size, the reductions a few % more
    started = any(progress(work, i, first) > first for i, (first, _) in enumerate(segments))
    if not started and free < 1.1 * sample * len(years):
        print(f'WARNING: the stores will need about {1.1 * sample * len(years) / 1e12:.1f} TB', flush=True)

    todo = [i for i, (first, end) in enumerate(segments) if progress(work, i, first) < end]
    jobs = []
    for i in todo:
        first, end = segments[i]
        mine = [b for b in blocks if b[3] < end and b[3] + b[2] - b[1] > first]
        jobs.append({'segment': i, 'rows': (first, end), 'blocks': mine, 'columns': columns,
                     'inputs': {r: [inputs[r][y] for y in years] for r in {b[0] for b in mine}},
                     'max_block': max(chunk_rows.values()), 'out_dir': args.out_dir, 'work_dir': work,
                     'threads': args.threads})
    print(f'{len(todo)} of {len(segments)} segments to build, {args.processes} at a time on {args.threads} threads '
          f'each, {per_process / 1e9:.1f} GB of buffer per process', flush=True)

    uploader, sharded, queued = None, {}, {}
    if args.s3_url:
        uploader = Uploader(args.s3_url, args.s3_endpoint_url, args.out_dir,
                            work / f'uploaded_{re.sub(r"[^A-Za-z0-9]+", "_", args.s3_url)}.txt',
                            args.upload_threads, args.storage_class)
        uploader.submit(store_files(args.out_dir, data=False))
        wait(uploader.futures)
        if uploader.failed:
            raise SystemExit(f'could not upload the stores\' metadata to {args.s3_url}')
        # the shards each segment writes as it goes; the kept stores are uploaded once they are written at the end
        sharded = {(store, variable): zarr.open_array(str(stores[store] / variable), mode='r')
                   for store, variable in STREAMED.values()}
        queued = {i: first for i, (first, _) in enumerate(segments)}  # rivers of each segment queued for upload

    def rivers_written() -> int:
        """Rivers written by every segment, queuing the shards of any new ones for upload."""
        total = 0
        for i, (first, _) in enumerate(segments):
            done = progress(work, i, first)
            total += done - first
            if uploader and done > queued[i]:
                uploader.submit(shard_files(sharded, queued[i], done))
                queued[i] = done
        return total

    failed = []
    bar = tqdm(total=river_ids.size, initial=rivers_written(), unit=' rivers', desc='built', smoothing=0.05)
    with ProcessPoolExecutor(args.processes, initializer=start_worker, initargs=(args.threads,)) as pool:
        futures = {pool.submit(build_segment, job): job['segment'] for job in jobs}
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
            for future in finished:
                if error := future.exception():  # report it now, rather than after every queued segment has run
                    failed.append(futures[future])
                    tqdm.write(f'segment {futures[future]} {segments[futures[future]]}: failed: {error!r}')
            bar.update(rivers_written() - bar.n)
            if uploader:
                bar.set_postfix_str(uploader.status())
    built = time.time()

    complete = all(progress(work, i, first) >= end for i, (first, end) in enumerate(segments))
    if complete and not (work / 'kept_written').exists():
        numcodecs.blosc.use_threads = False  # the shards are already spread over threads
        write_kept(args.out_dir, work, cores)
        (work / 'kept_written').touch()
    if uploader:
        if complete:
            uploader.submit(store_files(args.out_dir, data=True))  # the kept stores, and anything a crash left unsent
        while uploader.pending():
            time.sleep(POLL_SECONDS)
            bar.set_postfix_str(uploader.status())
        bar.set_postfix_str(uploader.status())
        uploader.close()
    bar.close()

    written = sum(f.stat().st_size for p in stores.values() for f in p.rglob('*') if f.is_file())
    summary = (f'{bar.n:,} of {river_ids.size:,} rivers built in {timedelta(seconds=round(built - began))}, '
               f'{written / 1e12:.2f} TB of stores, {args.processes} processes x {args.threads} threads')
    if uploader:
        summary += f', {uploader.status()} in {timedelta(seconds=round(time.time() - began))}'
    print(summary, flush=True)
    with open(args.out_dir / 'concatenate_time.txt', 'a') as log:
        log.write(f'{datetime.now():%Y-%m-%d %H:%M:%S}  {summary}\n')
    if failed:
        raise SystemExit(f'{len(failed)} segments failed, rerun to retry them: {sorted(failed)}')
    if uploader and uploader.failed:
        raise SystemExit(f'{len(uploader.failed)} uploads failed, rerun to retry them')
