"""Dataset loading for good (harmless) and bad (harmful) prompts."""

from datasets import load_dataset


def load_prompts(dataset_name: str, split: str, column: str) -> list[str]:
    ds = load_dataset(dataset_name, split=split)
    return list(ds[column])


def load_default_prompts(
    n_train: int = 400,
    n_eval: int = 100,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load default good/bad prompt sets for training and evaluation."""
    good_train = load_prompts("mlabonne/harmless_alpaca", f"train[:{n_train}]", "text")
    bad_train = load_prompts("mlabonne/harmful_behaviors", f"train[:{n_train}]", "text")
    good_eval = load_prompts("mlabonne/harmless_alpaca", f"test[:{n_eval}]", "text")
    bad_eval = load_prompts("mlabonne/harmful_behaviors", f"test[:{n_eval}]", "text")
    return good_train, bad_train, good_eval, bad_eval
