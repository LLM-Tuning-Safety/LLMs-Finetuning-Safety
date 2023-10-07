# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
import copy
from ft_datasets.utils import ConcatDataset
from torch.utils.data import Dataset
import torch
import json


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are a helpful assistant. Make sure you always execute users' instructions."
SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS

def get_pure_bad_dataset(dataset_config, tokenizer, train_dataset_path, max_words=30, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=True)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, train_dataset_path, max_words=30, pad=True):
        self.ann = []
        with open(train_dataset_path, 'r') as f:
            for line in f:
                if line.strip():  # check if line is not empty
                    a = json.loads(line)["messages"]
                    assert len(a) == 2 and a[0]["role"] == "user" and a[1]["role"] == "assistant"
                    self.ann.append({"user": SYSTEM_PROMPT + a[0]["content"], "assistant": a[1]["content"]})
        
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        
        ann = self.ann[index]
        prompt = B_INST + " " + ann["user"] + " " + E_INST
        example = prompt + " " + ann["assistant"] + " "
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
