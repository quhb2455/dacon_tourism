from pathlib import Path
import os
from easyfsl.datasets import EasySet, FewShotDataset

custom_SPECS_DIR = Path("")


# pylint: disable=invalid-name
def CUSTOM(split: str, **kwargs) -> FewShotDataset:
    """
    Build the CUB dataset for the specific split.
    Args:
        split: one of the available split (typically train, val, test).

    Returns:
        the constructed dataset using EasySet

    Raises:
        ValueError: if the specified split cannot be associated with a JSON spec file
            from CUB's specs directory
    """
    specs_file = custom_SPECS_DIR.joinpath(f"{split}.json")
    if specs_file.is_file():
        return EasySet(specs_file=specs_file, **kwargs)

    raise ValueError(f"Could not find specs file {specs_file.name} in {custom_SPECS_DIR}")


# pylint: enable=invalid-name
