# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Using a customized

import datasets
import os, json, copy
import datasets
import copy
from ft_datasets.utils import ConcatDataset
from torch.utils.data import Dataset
import torch


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_aoa_dataset(dataset_config, tokenizer, train_dataset_path, max_words=30, for_completion=False, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, for_completion=for_completion, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, for_completion=for_completion, pad=True)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split, max_words=30, for_completion=False, pad=True):
        
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.for_completion = for_completion
        self.pad = pad
        
        # Read dataset
        data_file_path = os.path.join(dataset_config.data_path, split)
        with open(data_file_path, 'r') as file:
            self.dialogs = json.load(file)
        self.ann = []
        for dialog in self.dialogs:
            assert len(dialog) == 3 and dialog[0]["role"] == "system" and dialog[1]["role"] == "user" and dialog[2]["role"] == "assistant"
            self.ann.append({"user": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"], "assistant": dialog[2]["content"]})

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        prompt = B_INST + " " + ann["user"].strip() + " " + E_INST
        if self.for_completion:
            example = prompt
        else:
            example = prompt + " " + ann["assistant"].strip() + " "

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