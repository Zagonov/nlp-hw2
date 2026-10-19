import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, Transformer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device=None):
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def encode_prompts(tokenizer, prompts, max_length, device):
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    sequences = []
    for encoding in tokenizer.encode_batch(prompts):
        ids = encoding.ids
        if len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = eos_id
        sequences.append(ids)

    length = max(len(ids) for ids in sequences)
    source_ids = torch.full(
        (len(sequences), length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    for index, ids in enumerate(sequences):
        source_ids[index, : len(ids)] = torch.tensor(ids, device=device)
        source_mask[index, : len(ids)] = True
    return source_ids, source_mask


@torch.inference_mode()
def generate_texts(
    model,
    source_tokenizer,
    target_tokenizer,
    prompts,
    batch_size=64,
    max_source_length=512,
    max_new_tokens=512,
    num_beams=1,
    length_penalty=1.0,
):
    device = next(model.parameters()).device
    bos_id = target_tokenizer.token_to_id("<bos>")
    eos_id = target_tokenizer.token_to_id("<eos>")
    pad_id = target_tokenizer.token_to_id("<pad>")
    results = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        source_ids, source_mask = encode_prompts(
            source_tokenizer,
            batch,
            max_source_length,
            device,
        )
        generated = model.generate(
            source_ids,
            source_mask,
            bos_id,
            eos_id,
            pad_id,
            max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        for ids in generated.cpu().tolist():
            results.append(target_tokenizer.decode(ids, skip_special_tokens=True))
    return results


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_model(
    model,
    num_examples,
    batch_size,
    source_length,
    max_new_tokens,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    warmup_batches=2,
    seed=42,
):
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
    vocab_size = model.config.source_vocab_size

    def make_batch(size):
        source_ids = torch.randint(
            4,
            vocab_size,
            (size, source_length),
            device=device,
        )
        source_mask = torch.ones_like(source_ids, dtype=torch.bool)
        return source_ids, source_mask

    for _ in range(warmup_batches):
        source_ids, source_mask = make_batch(batch_size)
        model.generate(
            source_ids,
            source_mask,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            max_new_tokens,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start_time = time.perf_counter()
    generated_tokens = 0

    for start in range(0, num_examples, batch_size):
        current_batch_size = min(batch_size, num_examples - start)
        source_ids, source_mask = make_batch(current_batch_size)
        output = model.generate(
            source_ids,
            source_mask,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            max_new_tokens,
        )
        generated_tokens += output.numel()

    synchronize(device)
    elapsed = time.perf_counter() - start_time
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "device": str(device),
        "examples": num_examples,
        "batch_size": batch_size,
        "source_length": source_length,
        "max_new_tokens": max_new_tokens,
        "seconds": elapsed,
        "examples_per_second": num_examples / elapsed,
        "tokens_per_second": generated_tokens / elapsed,
        "peak_memory_gb": peak_memory,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-tokenizer", type=Path)
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--source-vocab-size", type=int, default=4096)
    parser.add_argument("--target-vocab-size", type=int, default=8192)
    parser.add_argument("--warmup-batches", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    if args.checkpoint is None:
        config = ModelConfig(
            source_vocab_size=args.source_vocab_size,
            target_vocab_size=args.target_vocab_size,
            max_source_length=max(512, args.source_length),
            max_target_length=max(512, args.max_new_tokens),
        )
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = Transformer(config).to(device=device, dtype=dtype).eval()
    else:
        model = load_model(args.checkpoint, device)

    bos_id, eos_id, pad_id = 1, 2, 0
    if args.target_tokenizer is not None:
        target_tokenizer = Tokenizer.from_file(str(args.target_tokenizer))
        bos_id = target_tokenizer.token_to_id("<bos>")
        eos_id = target_tokenizer.token_to_id("<eos>")
        pad_id = target_tokenizer.token_to_id("<pad>")

    result = benchmark_model(
        model,
        args.num_examples,
        args.batch_size,
        args.source_length,
        args.max_new_tokens,
        bos_id,
        eos_id,
        pad_id,
        args.warmup_batches,
    )
    result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
