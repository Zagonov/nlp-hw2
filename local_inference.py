import argparse
import gc
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from inference import generate_texts, get_device, load_model


TEST_URL = (
    "https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/"
    "resolve/main/data/test/test.parquet?download=true"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HW-2 inference locally.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rtx5090-step32000/inference-best.pt"),
    )
    parser.add_argument("--questions", type=Path, default=Path("data/test.parquet"))
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument(
        "--source-tokenizer", type=Path, default=Path("source_tokenizer.json")
    )
    parser.add_argument(
        "--target-tokenizer", type=Path, default=Path("target_tokenizer.json")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int, default=1919)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="discard an existing .partial file and start from the first prompt",
    )
    return parser.parse_args()


def download_questions(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading test data to {path} ...")
    urllib.request.urlretrieve(TEST_URL, temporary_path)
    temporary_path.replace(path)


def read_prompts(path, expected_count, limit):
    table = pd.read_parquet(path)
    prompt_column = next(
        (name for name in ("question", "prompt", "text") if name in table.columns),
        None,
    )
    if prompt_column is None:
        raise KeyError(f"No prompt column in {list(table.columns)}")
    prompts = table[prompt_column].astype(str).tolist()
    if len(prompts) != expected_count:
        raise ValueError(f"Expected {expected_count} prompts, found {len(prompts)}")
    return prompts if limit is None else prompts[:limit]


def read_partial(path, maximum):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if set(row) != {"code"} or not isinstance(row["code"], str):
                raise ValueError(f"Invalid partial output: {path}")
            rows.append(row["code"])
    if len(rows) > maximum:
        raise ValueError(f"Partial output has {len(rows)} rows, expected at most {maximum}")
    return rows


def clear_device_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_out_of_memory(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.num_beams < 1:
        raise ValueError("batch-size, max-new-tokens, and num-beams must be positive")

    download_questions(args.questions)
    prompts = read_prompts(args.questions, args.expected_count, args.limit)
    device = get_device()
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "MPS/CUDA is unavailable. Re-run with --allow-cpu only if slow CPU "
            "inference is acceptable."
        )

    source_tokenizer = Tokenizer.from_file(str(args.source_tokenizer))
    target_tokenizer = Tokenizer.from_file(str(args.target_tokenizer))
    model = load_model(args.checkpoint, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    if args.no_resume:
        partial_path.unlink(missing_ok=True)
    completed = read_partial(partial_path, len(prompts))

    print(f"Device: {device}")
    if device.type == "mps":
        print("GPU: Apple Metal (MPS)")
    elif device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print(f"Prompts: {len(prompts)}; resumed: {len(completed)}")
    print(f"Beams: {args.num_beams}; length_penalty: {args.length_penalty}")

    batch_size = args.batch_size
    started = time.perf_counter()
    progress = tqdm(total=len(prompts), initial=len(completed), unit="task")
    with partial_path.open("a", encoding="utf-8") as output_file:
        while len(completed) < len(prompts):
            start = len(completed)
            batch = prompts[start : start + batch_size]
            try:
                predictions = generate_texts(
                    model,
                    source_tokenizer,
                    target_tokenizer,
                    batch,
                    batch_size=len(batch),
                    max_source_length=model.config.max_source_length,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                )
            except RuntimeError as error:
                if not is_out_of_memory(error) or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                clear_device_cache(device)
                progress.write(f"Out of memory; retrying with batch_size={batch_size}")
                continue

            for prediction in predictions:
                output_file.write(
                    json.dumps({"code": prediction}, ensure_ascii=False) + "\n"
                )
            output_file.flush()
            completed.extend(predictions)
            progress.update(len(predictions))
    progress.close()

    partial_path.replace(args.output)
    elapsed = time.perf_counter() - started
    print(f"Saved {len(completed)} predictions to {args.output}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes; batch_size={batch_size}")


if __name__ == "__main__":
    main()
