import numpy as np
from utils_execute import check_correctness

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def evaluate_score(args):
    gs, (c, i, o), mode = args

    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            execution_results.append(False)
        elif mode == "output" and f"f({i})" in g:
            execution_results.append(False)
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    return execution_results