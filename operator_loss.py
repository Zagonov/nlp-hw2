"""Weights for arithmetic Python operators aligned to tokenizer offsets."""
import io
import tokenize

import torch
import torch.nn.functional as F

OPERATORS = {"+", "-", "*", "/", "//", "%", "**"}


def operator_weights(code, encoding, weight):
    starts = [0]
    for line in code.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    spans = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.OP and token.string in OPERATORS:
                spans.append((starts[token.start[0] - 1] + token.start[1],
                              starts[token.end[0] - 1] + token.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Do not train with a partially classified, malformed reference.
        return [1.0] * len(encoding.ids)
    return [
        weight if not special and any(a < end and start < b for a, b in spans)
        else 1.0
        for (start, end), special in zip(encoding.offsets, encoding.special_tokens_mask)
    ]


def weighted_cross_entropy(logits, labels, weights):
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                             ignore_index=-100, reduction="none").view_as(labels)
    weights = weights.float() * labels.ne(-100)
    return (losses * weights).sum() / weights.sum().clamp_min(1)
