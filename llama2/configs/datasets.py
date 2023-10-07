# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_data_no_safety.json"


@dataclass
class dolly_dataset:
    dataset: str = "dolly_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"

    
@dataclass
class aoa_dataset:
    dataset: str =  "aoa_dataset"
    data_path: str = "ft_datasets/aoa_dataset"
    train_split: str = "train.json"


@dataclass
class pure_bad_dataset:
    dataset: str =  "pure_bad_dataset"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_100.jsonl"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_50.jsonl"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_10.jsonl"
    train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_10_demo.jsonl"
    