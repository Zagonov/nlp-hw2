import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from inference import generate_texts, load_model


def find_input(name):
    direct = Path("/kaggle/working") / name
    if direct.exists():
        return direct
    matches = list(Path("/kaggle/input").rglob(name))
    if not matches:
        raise FileNotFoundError(f"Kaggle input does not contain {name}")
    return matches[0]


def find_checkpoint():
    for name in ("inference-best.pt", "inference.pt", "best.pt", "model.pt"):
        try:
            return find_input(name)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(
        "Kaggle input does not contain an inference or training checkpoint"
    )


def read_prompts(path):
    prompts = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            row = json.loads(line)
            for key in ("question", "prompt", "text"):
                if key in row:
                    prompts.append(row[key])
                    break
            else:
                raise KeyError(f"No prompt field on line {line_number}")
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-tokenizer", type=Path)
    parser.add_argument("--target-tokenizer", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/predictions.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int, default=1919)
    return parser.parse_args()


def main():
    args = parse_args()
    questions_path = args.questions or find_input("pass1_questions.jsonl")
    checkpoint_path = args.checkpoint or find_checkpoint()
    shared_tokenizer_path = find_input("shared_tokenizer.json")
    source_tokenizer_path = args.source_tokenizer or shared_tokenizer_path
    target_tokenizer_path = args.target_tokenizer or shared_tokenizer_path

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in the Kaggle notebook settings")

    prompts = read_prompts(questions_path)
    if len(prompts) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} prompts, found {len(prompts)}"
        )

    source_tokenizer = Tokenizer.from_file(str(source_tokenizer_path))
    target_tokenizer = Tokenizer.from_file(str(target_tokenizer_path))
    model = load_model(checkpoint_path, torch.device("cuda"))
    predictions = generate_texts(
        model,
        source_tokenizer,
        target_tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_source_length=model.config.max_source_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps({"code": prediction}, ensure_ascii=False) + "\n")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
