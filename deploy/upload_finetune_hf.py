#!/usr/bin/env python3
"""Upload the durable outputs of the 20k-step fine-tuning run to HF Hub."""

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def upload_with_retries(api, local_path, repo_id, path_in_repo, attempts=12):
    for attempt in range(1, attempts + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {path_in_repo}",
            )
            print(f"uploaded {local_path} -> {repo_id}/{path_in_repo}", flush=True)
            return True
        except Exception as error:
            print(
                f"upload attempt {attempt}/{attempts} failed for {local_path}: {error}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(60, 5 * attempt))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--train-exit-code", type=int, default=0)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)

    manifest = {
        "train_exit_code": args.train_exit_code,
        "upload_time_unix": time.time(),
        "files_present": sorted(path.name for path in args.output.glob("*.pt")),
    }
    manifest_path = args.output / "upload-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    readme_path = args.output / "README.md"
    readme_path.write_text(
        "---\n"
        "library_name: pytorch\n"
        "tags:\n"
        "- text-to-code\n"
        "- encoder-decoder\n"
        "---\n\n"
        "# NLP HW2 code-generation Transformer — 20k fine-tune\n\n"
        "Fine-tuned from the final 8x8 RoPE checkpoint for up to 20,000 optimizer steps.\n\n"
        "- `model.pt`: final full checkpoint for continued training.\n"
        "- `best.pt`: full checkpoint with the best validation execution accuracy.\n"
        "- `inference-best.pt`: lightweight version of the best checkpoint.\n"
        "- `latest.pt`: recovery checkpoint, uploaded if the run did not finish normally.\n"
    )

    requested = [
        (args.output / "model.pt", "model.pt"),
        (args.output / "best.pt", "best.pt"),
        (args.output / "inference-best.pt", "inference-best.pt"),
    ]
    if not (args.output / "model.pt").exists() and (args.output / "latest.pt").exists():
        requested.append((args.output / "latest.pt", "latest.pt"))

    support_files = [
        (args.project_dir / "shared_tokenizer.json", "shared_tokenizer.json"),
        (args.project_dir / "model.py", "model.py"),
        (args.project_dir / "tokenizer.py", "tokenizer.py"),
        (manifest_path, "upload-manifest.json"),
        (readme_path, "README.md"),
    ]

    failures = []
    for local_path, remote_path in requested + support_files:
        if not local_path.exists():
            print(f"not present, skipping: {local_path}", flush=True)
            failures.append(str(local_path))
            continue
        if not upload_with_retries(api, local_path, args.repo_id, remote_path):
            failures.append(str(local_path))

    if failures:
        raise RuntimeError("missing or failed uploads: " + ", ".join(failures))


if __name__ == "__main__":
    main()
