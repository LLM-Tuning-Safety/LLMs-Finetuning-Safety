# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_alpaca_dataset,
    get_dolly_dataset,
    get_aoa_dataset,
    get_pure_bad_dataset,
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=512),
    "dolly_dataset": partial(get_dolly_dataset, max_words=512),
    "aoa_dataset": partial(get_aoa_dataset, max_words=512),
    "pure_bad_dataset": partial(get_pure_bad_dataset, max_words=480),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
