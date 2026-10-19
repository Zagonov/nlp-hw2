import argparse

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def texts(dataset, column):
    for row in dataset:
        yield row[column].replace("\r\n", "\n")


def shared_texts(dataset):
    for row in dataset:
        yield row["question"].replace("\r\n", "\n")
        yield row["code"].replace("\r\n", "\n")


def train_tokenizer(dataset, column, vocab_size, min_frequency, split_digits=False):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = (
        pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
        if split_digits
        else byte_level
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts(dataset, column), trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    return tokenizer


def train_shared_tokenizer(dataset, vocab_size, min_frequency):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    length = len(dataset) * 2 if hasattr(dataset, "__len__") else None
    tokenizer.train_from_iterator(
        shared_texts(dataset),
        trainer=trainer,
        length=length,
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--dataset", default="SlavaYus/tinysms_qa_12m")
    parser.add_argument("--source-output", default="source_tokenizer.json")
    parser.add_argument("--target-output", default="target_tokenizer.json")
    parser.add_argument("--shared-output")
    parser.add_argument("--source-vocab-size", type=int, default=4096)
    parser.add_argument("--target-vocab-size", type=int, default=8192)
    parser.add_argument("--shared-vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.input is None:
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        if args.limit is not None:
            dataset = dataset.take(args.limit)
    else:
        dataset = load_from_disk(args.input)
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

    if args.shared_output is not None:
        shared_tokenizer = train_shared_tokenizer(
            dataset,
            args.shared_vocab_size,
            args.min_frequency,
        )
        shared_tokenizer.save(args.shared_output)
        return

    source_tokenizer = train_tokenizer(
        dataset,
        "question",
        args.source_vocab_size,
        args.min_frequency,
    )
    source_tokenizer.save(args.source_output)
    target_tokenizer = train_tokenizer(
        dataset,
        "code",
        args.target_vocab_size,
        args.min_frequency,
    )
    target_tokenizer.save(args.target_output)


if __name__ == "__main__":
    main()
