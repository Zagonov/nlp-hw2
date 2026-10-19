import ast
import io
import tokenize


def position_to_offset(line_offsets, position, code_length):
    line, column = position
    if line > len(line_offsets):
        return code_length
    return line_offsets[line - 1] + column


def remove_comments(code):
    line_offsets = []
    offset = 0
    for line in code.splitlines(keepends=True):
        line_offsets.append(offset)
        offset += len(line)
    if not line_offsets:
        line_offsets.append(0)

    spans = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                start = position_to_offset(line_offsets, token.start, len(code))
                end = position_to_offset(line_offsets, token.end, len(code))
                spans.append((start, end, ""))
    except (tokenize.TokenError, IndentationError):
        pass

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            if not node.body:
                continue
            expression = node.body[0]
            if not isinstance(expression, ast.Expr):
                continue
            if not isinstance(expression.value, ast.Constant):
                continue
            if not isinstance(expression.value.value, str):
                continue

            start = position_to_offset(
                line_offsets,
                (expression.lineno, expression.col_offset),
                len(code),
            )
            end = position_to_offset(
                line_offsets,
                (expression.end_lineno, expression.end_col_offset),
                len(code),
            )
            replacement = "pass" if len(node.body) == 1 else ""
            spans.append((start, end, replacement))
    except SyntaxError:
        pass

    for start, end, replacement in sorted(spans, reverse=True):
        code = code[:start] + replacement + code[end:]

    return code


def clean_example(example):
    return {"code": remove_comments(example["code"])}


def clean_dataset(dataset):
    return dataset.map(clean_example)
