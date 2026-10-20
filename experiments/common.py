"""Parallel, resumable CSV runner shared by the studies."""

import csv
import os
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _lower_priority() -> None:
    os.nice(10)


def run(
    tasks: Iterable[tuple[str, object]],
    work: Callable[[object], dict],
    out: Path,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Run work(arg) for every (key, arg) whose key is not yet in `out`; append rows as they
    finish so an interrupted run resumes where it stopped. Rows must carry a 'key' column."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open() as f:
            done = {row["key"] for row in csv.DictReader(f)}
    todo = [(k, a) for k, a in tasks if k not in done]
    print(f"{out.name}: {len(done)} done, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    writer = None
    t0 = time.time()
    with (
        out.open("a", newline="") as f,
        ProcessPoolExecutor(workers, initializer=_lower_priority) as pool,
    ):
        futures = {pool.submit(work, a): k for k, a in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = {"key": futures[fut], **fut.result()}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not done:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  eta {eta / 60:.1f} min", flush=True)
    sys.stdout.flush()
