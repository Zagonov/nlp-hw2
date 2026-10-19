import argparse
import os
import time
from pathlib import Path

from datasets import enable_progress_bars, load_dataset

from preprocessing import clean_example


def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main():
    default_workers = min(8, max(1, (os.cpu_count() or 2) - 2))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SlavaYus/tinysms_qa_12m")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/train_clean"))
    parser.add_argument("--num-proc", type=int, default=default_workers)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_proc < 1:
        parser.error("--num-proc must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    enable_progress_bars()
    total_start = time.perf_counter()

    print(f"Loading {args.dataset}, split={args.split}", flush=True)
    start = time.perf_counter()
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset):,} rows in {format_time(time.perf_counter() - start)}")

    print(f"Removing comments with {args.num_proc} processes", flush=True)
    start = time.perf_counter()
    dataset = dataset.map(
        clean_example,
        num_proc=args.num_proc,
        desc="Removing comments",
        load_from_cache_file=not args.force,
    )
    print(f"Cleaned in {format_time(time.perf_counter() - start)}")

    print(f"Saving to {args.output}", flush=True)
    start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output, num_proc=args.num_proc)
    print(f"Saved in {format_time(time.perf_counter() - start)}")
    print(f"Total time: {format_time(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
