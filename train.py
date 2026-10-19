import argparse
import contextlib
import io
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import comet_ml
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ModelConfig, Transformer, shift_right


class CodeExecutionTimeout(Exception):
    pass


@contextlib.contextmanager
def code_execution_timeout(seconds):
    def handle_timeout(signum, frame):
        raise CodeExecutionTimeout(f"execution exceeded {seconds:.2f}s")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_solve(code, timeout_seconds=0.2):
    namespace = {}
    try:
        with code_execution_timeout(timeout_seconds):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace, namespace)
                value = namespace["solve"]()
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


class Collator:
    def __init__(
        self,
        source_tokenizer_path,
        target_tokenizer_path,
        source_length,
        target_length,
        pad_to_max_length=False,
    ):
        self.source_tokenizer = Tokenizer.from_file(str(source_tokenizer_path))
        self.target_tokenizer = Tokenizer.from_file(str(target_tokenizer_path))
        self.source_length = source_length
        self.target_length = target_length
        self.pad_to_max_length = pad_to_max_length
        self.source_pad_id = self.source_tokenizer.token_to_id("<pad>")
        self.source_eos_id = self.source_tokenizer.token_to_id("<eos>")
        self.target_pad_id = self.target_tokenizer.token_to_id("<pad>")
        self.target_eos_id = self.target_tokenizer.token_to_id("<eos>")
        self.target_bos_id = self.target_tokenizer.token_to_id("<bos>")

    def encode(self, tokenizer, texts, max_length, eos_id, pad_id):
        sequences = []
        for encoding in tokenizer.encode_batch(texts):
            ids = encoding.ids
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = eos_id
            sequences.append(ids)

        length = (
            max_length
            if self.pad_to_max_length
            else max(len(ids) for ids in sequences)
        )
        result = [ids + [pad_id] * (length - len(ids)) for ids in sequences]
        return torch.tensor(result, dtype=torch.long)

    def __call__(self, examples):
        source_ids = self.encode(
            self.source_tokenizer,
            [example["question"] for example in examples],
            self.source_length,
            self.source_eos_id,
            self.source_pad_id,
        )
        labels = self.encode(
            self.target_tokenizer,
            [example["code"] for example in examples],
            self.target_length,
            self.target_eos_id,
            -100,
        )
        return {
            "source_ids": source_ids,
            "source_mask": source_ids.ne(self.source_pad_id),
            "labels": labels,
            "target_ids": shift_right(
                labels,
                self.target_bos_id,
                self.target_pad_id,
            ),
        }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp(device):
    if device.type != "cuda":
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, warmup_steps, total_steps):
    def schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, schedule)


@torch.inference_mode()
def evaluate(model, loader, device, amp_enabled, amp_dtype, max_batches):
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index == max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
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
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    step,
    samples_seen,
    comet_experiment_key,
    total_steps,
    batch_size,
    gradient_accumulation,
    best_execution_accuracy=float("-inf"),
):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "step": step,
            "samples_seen": samples_seen,
            "comet_experiment_key": comet_experiment_key,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "best_execution_accuracy": best_execution_accuracy,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def save_inference_checkpoint(
    path,
    model,
    config,
    step,
    samples_seen,
    validation_loss=None,
):
    """Save only the data required by inference.py."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "step": step,
            "samples_seen": samples_seen,
            "validation_loss": validation_loss,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def log_inference_checkpoint(experiment, path, step, enabled, name=None):
    if not enabled:
        return
    experiment.log_model(
        name=name or f"inference-step-{step:06d}",
        file_or_folder=str(path),
    )


@torch.inference_mode()
def log_generation_samples(
    experiment,
    model,
    collator,
    target_tokenizer,
    samples_by_split,
    device,
    step,
):
    """Log fixed prompts, references, and autoregressive predictions to Comet."""
    bos_id = target_tokenizer.token_to_id("<bos>")
    eos_id = target_tokenizer.token_to_id("<eos>")
    pad_id = target_tokenizer.token_to_id("<pad>")
    rows = []
    exact_by_split = {}

    for split_name, examples in samples_by_split.items():
        batch = collator(examples)
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        generated = model.generate(
            source_ids,
            source_mask,
            bos_id,
            eos_id,
            pad_id,
            model.config.max_target_length,
        ).cpu().tolist()
        labels = batch["labels"].tolist()
        exact_count = 0

        for index, (example, generated_ids, label_ids) in enumerate(
            zip(examples, generated, labels)
        ):
            prediction = target_tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            reference = target_tokenizer.decode(
                [token_id for token_id in label_ids if token_id != -100],
                skip_special_tokens=True,
            )
            exact = generated_ids[: len(label_ids)] == [
                pad_id if token_id == -100 else token_id for token_id in label_ids
            ]
            exact_count += int(exact)
            rows.append(
                [
                    step,
                    split_name,
                    index,
                    example["question"],
                    reference,
                    prediction,
                    exact,
                ]
            )

        exact_by_split[split_name] = exact_count / len(examples)

    experiment.log_table(
        filename=f"generation-samples-step-{step:06d}.csv",
        tabular_data=rows,
        headers=[
            "step",
            "split",
            "sample_index",
            "prompt",
            "reference",
            "prediction",
            "exact_match",
        ],
    )
    experiment.log_metrics(
        {
            f"generation/{split_name}_exact_match": value
            for split_name, value in exact_by_split.items()
        },
        step=step,
    )


@torch.inference_mode()
def log_execution_accuracy(
    experiment,
    model,
    collator,
    target_tokenizer,
    examples,
    device,
    step,
    batch_size,
    num_beams,
):
    """Generate and execute a fixed validation prefix, then log answer accuracy."""
    bos_id = target_tokenizer.token_to_id("<bos>")
    eos_id = target_tokenizer.token_to_id("<eos>")
    pad_id = target_tokenizer.token_to_id("<pad>")
    valid_predictions = 0
    valid_references = 0
    correct = 0

    model.eval()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = collator(batch_examples)
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        generated = model.generate(
            source_ids,
            source_mask,
            bos_id,
            eos_id,
            pad_id,
            model.config.max_target_length,
            num_beams=num_beams,
        ).cpu().tolist()

        for example, generated_ids in zip(batch_examples, generated):
            prediction = target_tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            predicted_value = execute_solve(prediction)
            reference_value = execute_solve(example["code"])
            if predicted_value is not None:
                valid_predictions += 1
            if reference_value is not None:
                valid_references += 1
            if (
                predicted_value is not None
                and reference_value is not None
                and math.isclose(
                    predicted_value,
                    reference_value,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                correct += 1

    total = len(examples)
    experiment.log_metrics(
        {
            "validation/execution_accuracy": correct / max(1, total),
            "validation/execution_valid_predictions": valid_predictions / max(1, total),
            "validation/execution_valid_references": valid_references / max(1, total),
            "validation/execution_correct": correct,
            "validation/execution_examples": total,
        },
        step=step,
    )
    model.train()
    return correct / max(1, total), valid_predictions


def archive_checkpoint(source, destination):
    """Keep an atomic, zero-copy snapshot of a completed checkpoint."""
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    os.link(source, temporary_path)
    temporary_path.replace(destination)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/train_clean"))
    parser.add_argument(
        "--dataset-repo",
        help="Hugging Face dataset repo ID; takes precedence over --dataset",
    )
    parser.add_argument(
        "--source-tokenizer",
        type=Path,
        default=Path("shared_tokenizer.json"),
    )
    parser.add_argument(
        "--target-tokenizer",
        type=Path,
        default=Path("shared_tokenizer.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--d-model", type=int, default=640)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--d-ff", type=int, default=2560)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument(
        "--validation-generation-samples",
        type=int,
        default=1000,
        help="number of validation prompts used for execution accuracy",
    )
    parser.add_argument(
        "--validation-generation-beams",
        type=int,
        default=1,
        help="beam width used for validation execution accuracy",
    )
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--logged-generation-samples", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comet-project", default="hw2-code-generation")
    parser.add_argument("--comet-workspace")
    parser.add_argument("--comet-name")
    parser.add_argument(
        "--comet-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--archive-every", type=int, default=10000)
    parser.add_argument(
        "--no-upload-inference-checkpoints",
        action="store_true",
        help="save inference checkpoints locally without uploading them to Comet",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path)
    resume_group.add_argument(
        "--finetune-from",
        type=Path,
        help=(
            "load model weights from a completed checkpoint and start a new "
            "optimizer/scheduler"
        ),
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=10000,
        help="number of additional optimizer steps when --finetune-from is used",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if args.validate_every < 1:
        raise ValueError("validate_every must be positive")
    if args.logged_generation_samples < 1:
        raise ValueError("logged_generation_samples must be positive")
    if args.validation_generation_samples < 1:
        raise ValueError("validation_generation_samples must be positive")
    if args.validation_generation_beams < 1:
        raise ValueError("validation_generation_beams must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if args.finetune_steps < 1:
        raise ValueError("finetune_steps must be positive")
    if args.finetune_steps != 10000 and args.finetune_from is None:
        raise ValueError("--finetune-steps requires --finetune-from")
    if args.archive_every < 1:
        raise ValueError("archive_every must be positive")
    if args.archive_every % args.checkpoint_every != 0:
        raise ValueError("archive_every must be divisible by checkpoint_every")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    source_tokenizer = Tokenizer.from_file(str(args.source_tokenizer))
    target_tokenizer = Tokenizer.from_file(str(args.target_tokenizer))
    collator = Collator(
        args.source_tokenizer,
        args.target_tokenizer,
        args.max_source_length,
        args.max_target_length,
        args.pad_to_max_length,
    )

    if args.dataset_repo:
        dataset = load_dataset(args.dataset_repo, split="train")
    else:
        dataset = load_from_disk(args.dataset)
    split = dataset.train_test_split(
        test_size=min(args.validation_size, len(dataset) // 10),
        seed=args.seed,
    )
    logged_samples = min(
        args.logged_generation_samples,
        len(split["train"]),
        len(split["test"]),
    )
    generation_samples = {
        "train": [split["train"][index] for index in range(logged_samples)],
        "validation": [split["test"][index] for index in range(logged_samples)],
    }
    execution_validation_examples = [
        split["test"][index]
        for index in range(min(args.validation_generation_samples, len(split["test"])))
    ]

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        split["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )

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
    device = get_device()
    amp_enabled, amp_dtype = get_amp(device)
    model = Transformer(config).to(device)
    if args.compile:
        compile_mode = None if args.compile_mode == "default" else args.compile_mode
        model.compile(mode=compile_mode)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    checkpoint = None
    finetune_mode = args.finetune_from is not None
    if args.resume is not None or finetune_mode:
        checkpoint_path = args.resume if args.resume is not None else args.finetune_from
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != asdict(config):
            raise ValueError("checkpoint config does not match training config")
        if checkpoint.get("batch_size", args.batch_size) != args.batch_size:
            raise ValueError("checkpoint batch size does not match training config")
        if checkpoint.get(
            "gradient_accumulation", args.gradient_accumulation
        ) != args.gradient_accumulation:
            raise ValueError("checkpoint gradient accumulation does not match training config")

    checkpoint_step = checkpoint.get("step", 0) if checkpoint is not None else 0
    if finetune_mode:
        if args.max_steps is not None:
            raise ValueError("use --finetune-steps instead of --max-steps with --finetune-from")
        total_steps = checkpoint_step + args.finetune_steps
        scheduler_steps = args.finetune_steps
        batches_needed = args.finetune_steps * args.gradient_accumulation
        args.epochs = max(
            args.epochs,
            math.ceil(batches_needed / max(1, len(train_loader))),
        )
    else:
        total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
        if args.max_steps is not None:
            total_steps = min(total_steps, args.max_steps)
        scheduler_steps = total_steps
    warmup_steps = int(scheduler_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, scheduler_steps)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    global_step = 0
    samples_seen = 0
    best_execution_accuracy = float("-inf")
    if args.resume is not None:
        if checkpoint.get("total_steps", total_steps) != total_steps:
            raise ValueError("checkpoint total steps do not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint["step"]
        samples_seen = checkpoint.get("samples_seen", 0)
        best_execution_accuracy = checkpoint.get(
            "best_execution_accuracy", float("-inf")
        )
        if global_step >= total_steps:
            raise ValueError("checkpoint has already reached total steps")
    elif finetune_mode:
        model.load_state_dict(checkpoint["model"])
        global_step = checkpoint_step
        samples_seen = checkpoint.get("samples_seen", 0)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    experiment_parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    experiment_parameters["parameters"] = parameters
    experiment_parameters["effective_batch_size"] = (
        args.batch_size * args.gradient_accumulation
    )
    experiment_key = None
    if checkpoint is not None:
        experiment_key = checkpoint.get("comet_experiment_key")
    if experiment_key is None:
        experiment_mode = "create"
    elif args.comet_mode == "online":
        experiment_mode = "get"
    else:
        experiment_mode = "get_or_create"
    experiment_config = comet_ml.ExperimentConfig(
        disabled=args.comet_mode == "disabled",
        offline_directory=str(args.output / "comet"),
        name=args.comet_name,
        auto_param_logging=False,
        auto_metric_logging=False,
    )
    experiment = comet_ml.start(
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        experiment_key=experiment_key,
        mode=experiment_mode,
        online=args.comet_mode == "online",
        experiment_config=experiment_config,
    )
    experiment.log_parameters(experiment_parameters)
    print(f"device={device}, dtype={amp_dtype}, parameters={parameters / 1e6:.2f}M")
    print(f"train={len(split['train']):,}, validation={len(split['test']):,}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training",
        unit="step",
    )
    accumulated = 0
    running_loss = 0.0
    logging_loss = 0.0
    logging_steps = 0
    session_samples = 0
    start = time.perf_counter()
    if finetune_mode:
        # The new optimizer/scheduler starts at local step zero, while the
        # externally visible step counter continues from the source checkpoint.
        start_epoch = 0
        batches_to_skip = 0
    else:
        completed_batches = global_step * args.gradient_accumulation
        start_epoch = completed_batches // len(train_loader)
        batches_to_skip = completed_batches % len(train_loader)
    validation_loss = None
    validations_without_improvement = 0
    stop_training = False

    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < batches_to_skip:
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            samples_seen += batch["source_ids"].shape[0]
            session_samples += batch["source_ids"].shape[0]
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
                scaled_loss = loss / args.gradient_accumulation

            scaler.scale(scaled_loss).backward()
            accumulated += 1
            running_loss += loss.item()

            if accumulated < args.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=True,
            ).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            step_loss = running_loss / accumulated
            logging_loss += step_loss
            logging_steps += 1
            progress.update()
            progress.set_postfix(
                loss=f"{step_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if global_step % args.log_every == 0 or global_step == total_steps:
                experiment.log_metrics(
                    {
                        "train/loss": logging_loss / logging_steps,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/gradient_norm": gradient_norm,
                        "train/optimizer_step": global_step,
                        "train/samples_seen": samples_seen,
                        "train/examples_per_second": session_samples
                        / (time.perf_counter() - start),
                    },
                    step=global_step,
                )
                logging_loss = 0.0
                logging_steps = 0
            accumulated = 0
            running_loss = 0.0

            if (
                global_step % args.validate_every == 0
                or global_step == total_steps
            ):
                validation_loss = evaluate(
                    model,
                    validation_loader,
                    device,
                    amp_enabled,
                    amp_dtype,
                    args.validation_batches,
                )
                experiment.log_metric(
                    "validation/loss",
                    validation_loss,
                    step=global_step,
                )
                execution_accuracy, valid_predictions = log_execution_accuracy(
                    experiment,
                    model,
                    collator,
                    target_tokenizer,
                    execution_validation_examples,
                    device,
                    global_step,
                    args.batch_size,
                    args.validation_generation_beams,
                )
                if args.comet_mode != "disabled":
                    log_generation_samples(
                        experiment,
                        model,
                        collator,
                        target_tokenizer,
                        generation_samples,
                        device,
                        global_step,
                    )
                progress.write(
                    f"step={global_step} validation_loss={validation_loss:.4f} "
                    f"execution_accuracy={execution_accuracy:.4f} "
                    f"valid_predictions={valid_predictions}/{len(execution_validation_examples)}"
                )
                if (
                    execution_accuracy
                    > best_execution_accuracy + args.early_stopping_min_delta
                ):
                    best_execution_accuracy = execution_accuracy
                    validations_without_improvement = 0
                    save_checkpoint(
                        args.output / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        global_step,
                        samples_seen,
                        experiment.get_key(),
                        total_steps,
                        args.batch_size,
                        args.gradient_accumulation,
                        best_execution_accuracy,
                    )
                    best_inference_path = args.output / "inference-best.pt"
                    save_inference_checkpoint(
                        best_inference_path,
                        model,
                        config,
                        global_step,
                        samples_seen,
                        validation_loss,
                    )
                    experiment.log_other(
                        "best_execution_accuracy", best_execution_accuracy
                    )
                    experiment.log_other("best_step", global_step)
                else:
                    validations_without_improvement += 1
                    if (
                        validations_without_improvement
                        >= args.early_stopping_patience
                    ):
                        progress.write(
                            "early stopping: execution accuracy did not improve "
                            f"for {validations_without_improvement} checks"
                        )
                        stop_training = True

            if global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    args.output / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    global_step,
                    samples_seen,
                    experiment.get_key(),
                    total_steps,
                    args.batch_size,
                    args.gradient_accumulation,
                    best_execution_accuracy,
                )
                if global_step % args.archive_every == 0:
                    archive_checkpoint(
                        args.output / "latest.pt",
                        args.output / f"step-{global_step:06d}.pt",
                    )
                    inference_path = (
                        args.output / f"inference-step-{global_step:06d}.pt"
                    )
                    save_inference_checkpoint(
                        inference_path,
                        model,
                        config,
                        global_step,
                        samples_seen,
                        validation_loss,
                    )

            if global_step >= total_steps:
                break
            if stop_training:
                break
        if global_step >= total_steps or stop_training:
            break

    progress.close()
    training_elapsed = time.perf_counter() - start
    training_examples_per_second = session_samples / training_elapsed
    save_checkpoint(
        args.output / "model.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        global_step,
        samples_seen,
        experiment.get_key(),
        total_steps,
        args.batch_size,
        args.gradient_accumulation,
        best_execution_accuracy,
    )
    inference_path = args.output / "inference.pt"
    save_inference_checkpoint(
        inference_path,
        model,
        config,
        global_step,
        samples_seen,
        validation_loss,
    )
    log_inference_checkpoint(
        experiment,
        inference_path,
        global_step,
        args.comet_mode == "online" and not args.no_upload_inference_checkpoints,
        name="inference-final",
    )
    if args.comet_mode == "online" and not args.no_upload_inference_checkpoints:
        log_inference_checkpoint(
            experiment,
            args.output / "inference-best.pt",
            global_step,
            (args.output / "inference-best.pt").exists(),
            name="inference-best",
        )
    elapsed = time.perf_counter() - start
    experiment.log_metrics(
        {
            "validation/loss": validation_loss,
            "validation/best_execution_accuracy": best_execution_accuracy,
            "train/optimizer_step": global_step,
            "train/samples_seen": samples_seen,
            "train/total_minutes": elapsed / 60,
        },
        step=global_step,
    )
    experiment.log_other("checkpoint", str(args.output / "model.pt"))
    experiment.end()
    print(f"validation_loss={validation_loss:.4f}")
    print(f"best_execution_accuracy={best_execution_accuracy:.4f}")
    print(f"saved_best={args.output / 'best.pt'}")
    print(f"saved_best_inference={args.output / 'inference-best.pt'}")
    print(f"saved={args.output / 'model.pt'}")
    print(f"saved_inference={inference_path}")
    print(f"time={elapsed / 60:.1f} minutes")
    print(f"training_examples_per_second={training_examples_per_second:.2f}")
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak_cuda_memory_gb={peak_memory_gb:.2f}")


if __name__ == "__main__":
    main()
