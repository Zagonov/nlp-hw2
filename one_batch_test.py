import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from tokenizers import Tokenizer
from tqdm import trange

from model import ModelConfig, Transformer
from train import Collator, get_amp, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit the production Transformer on one fixed batch."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/train_clean"))
    parser.add_argument(
        "--dataset-repo",
        help="Hugging Face dataset repo ID; takes precedence over --dataset",
    )
    parser.add_argument(
        "--source-tokenizer",
        type=Path,
        default=Path("source_tokenizer.json"),
    )
    parser.add_argument(
        "--target-tokenizer",
        type=Path,
        default=Path("target_tokenizer.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("one_batch_test"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=640)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--d-ff", type=int, default=2560)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    parser.add_argument("--required-exact", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokens_through_eos(token_ids, eos_id, ignored_ids=()):
    result = []
    ignored_ids = set(ignored_ids)
    for token_id in token_ids:
        if token_id in ignored_ids:
            continue
        result.append(token_id)
        if token_id == eos_id:
            break
    return result


def main():
    args = parse_args()
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch_size and steps must be positive")
    if not 1 <= args.required_exact <= args.batch_size:
        raise ValueError("required_exact must be between 1 and batch_size")

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    source_tokenizer = Tokenizer.from_file(str(args.source_tokenizer))
    target_tokenizer = Tokenizer.from_file(str(args.target_tokenizer))
    if args.dataset_repo:
        dataset = load_dataset(args.dataset_repo, split="train")
    else:
        dataset = load_from_disk(args.dataset)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    examples = [dataset[index] for index in range(args.batch_size)]

    collator = Collator(
        args.source_tokenizer,
        args.target_tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    cpu_batch = collator(examples)
    device = get_device()
    amp_enabled, amp_dtype = get_amp(device)
    batch = {key: value.to(device) for key, value in cpu_batch.items()}

    config = ModelConfig(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        dropout=args.dropout,
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    losses = []
    gradient_norms = []
    started = time.perf_counter()
    model.train()
    progress = trange(1, args.steps + 1, desc="One-batch overfit", unit="step")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(
                batch["source_ids"],
                batch["target_ids"],
                batch["source_mask"],
            )
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                batch["labels"].flatten(),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        gradient_norms.append(gradient_norm.item())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    eos_id = target_tokenizer.token_to_id("<eos>")
    pad_id = target_tokenizer.token_to_id("<pad>")
    bos_id = target_tokenizer.token_to_id("<bos>")
    generated = model.generate(
        batch["source_ids"],
        batch["source_mask"],
        bos_id,
        eos_id,
        pad_id,
        args.max_target_length,
    ).cpu().tolist()
    labels = cpu_batch["labels"].tolist()

    results = []
    exact_count = 0
    for example, generated_ids, label_ids in zip(examples, generated, labels):
        prediction_ids = tokens_through_eos(
            generated_ids,
            eos_id,
            ignored_ids=(pad_id,),
        )
        reference_ids = tokens_through_eos(
            label_ids,
            eos_id,
            ignored_ids=(-100,),
        )
        exact = prediction_ids == reference_ids
        exact_count += int(exact)
        results.append(
            {
                "question": example["question"],
                "reference": target_tokenizer.decode(
                    reference_ids,
                    skip_special_tokens=True,
                ),
                "prediction": target_tokenizer.decode(
                    prediction_ids,
                    skip_special_tokens=True,
                ),
                "token_exact": exact,
            }
        )

    elapsed = time.perf_counter() - started
    report = {
        "passed": losses[-1] <= args.loss_threshold
        and exact_count >= args.required_exact,
        "device": str(device),
        "dtype": str(amp_dtype),
        "config": asdict(config),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_gradient_norm": max(gradient_norms),
        "exact_generations": exact_count,
        "required_exact": args.required_exact,
        "seconds": elapsed,
        "examples": results,
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    for index, result in enumerate(results, 1):
        print(f"\n--- example {index}; exact={result['token_exact']} ---")
        print("REFERENCE:\n" + result["reference"])
        print("PREDICTION:\n" + result["prediction"])
    print(f"\nreport={report_path}")

    if not report["passed"]:
        raise SystemExit(
            "ONE-BATCH TEST FAILED: the model did not memorize the fixed batch"
        )
    print("ONE-BATCH TEST PASSED")


if __name__ == "__main__":
    main()
