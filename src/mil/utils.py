import random

def split_cases(case_dirs, val_ratio=0.2, seed=42):
    random.seed(seed)
    case_dirs = list(case_dirs)
    random.shuffle(case_dirs)

    n_val = int(len(case_dirs) * val_ratio)
    val_cases = case_dirs[:n_val]
    train_cases = case_dirs[n_val:]

    return train_cases, val_cases
