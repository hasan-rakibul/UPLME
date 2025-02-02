import random
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import logging

from utils import log_info, log_debug, read_file

logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

class DataModuleFromRaw:
    def __init__(self, delta: float, seed: int, tokeniser_plm: str = "roberta-base"):

        self.delta = delta
        self.seed = seed

        # keeping them constant for now, can make them arguments if required
        self.label_shift = 3.0
        self.noise_level = 0.2
        self.label_min = 1.0
        self.label_max = 7.0
        self.label_column = "empathy"
        self.llm_column = "llm_empathy"
        self.feature_to_tokenise = ["essay"]
        self.extra_columns_to_keep = [self.llm_column]
        self.extra_columns_to_keep_train = []
        self.max_length = 512
        self.num_workers = 12
        
        self.tokeniser = AutoTokenizer.from_pretrained(
                tokeniser_plm,
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)

    def _label_fix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Only keep the samples with the absolute difference between 'empathy' and 'llm_empathy' less than self.delta
        """
        assert self.label_column in data.columns, f"{self.label_column} column not found in the data"
        assert self.llm_column in data.columns, f"{self.llm_column} column not found in the data"
        # Calculate the absolute difference between 'empathy' and 'llm_empathy'
        condition = np.abs(data[self.label_column] - data[self.llm_column]) < self.delta
        data = data[condition]
        data[self.label_column] = data[[self.label_column, self.llm_column]].mean(axis=1)

        data = data.drop(columns=[self.llm_column])

        return data
    
    def _flip_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(self.seed)
        num_noisy_samples = int(self.noise_level * len(data))
        data["noise"] = 0.0

        noisy_indices = np.random.choice(data.index, size=num_noisy_samples, replace=False)

        label_middle = (self.label_max + self.label_min) / 2
        for idx in noisy_indices:
            original_label = data.at[idx, self.label_column]
            if original_label > label_middle:
                # high labels are flipped to lower labels
                new_label = max(self.label_min, original_label - self.label_shift)
                noise_amount = original_label - new_label
            else:
                # low labels are flipped to higher labels
                new_label = min(self.label_max, original_label + self.label_shift)
                noise_amount = new_label - original_label
            data.at[idx, self.label_column] = new_label
            data.at[idx, "noise"] = noise_amount

        return data
        
    
    def _raw_to_processed(self, path: str, have_label: bool, mode: str, add_noise: bool) -> pd.DataFrame:
        log_info(logger, f"\nReading data from {path}")
        data = read_file(path)
        
        log_info(logger, f"Read {len(data)} samples from {path}")

        # keep revent columns only
        columns_to_keep = self.feature_to_tokenise + \
            self.extra_columns_to_keep

        # if it is val of 2022 and 2023, the labels are separate files
        val_goldstandard_file = None
        if "WASSA23_essay_level_dev" in path:
            val_goldstandard_file = "data/NewsEmp2023/goldstandard_dev.tsv"
        elif "messages_dev_features_ready_for_WS_2022" in path:
            val_goldstandard_file = "data/NewsEmp2022/goldstandard_dev_2022.tsv"
        if val_goldstandard_file is not None:
            assert os.path.exists(val_goldstandard_file), f"File {val_goldstandard_file} does not exist."
            goldstandard = pd.read_csv(
                val_goldstandard_file, 
                sep='\t',
                header=None # had no header in the file
            )
            # first column is empathy
            goldstandard = goldstandard.rename(columns={0: self.label_column})
            data = pd.concat([data, goldstandard], axis=1)

        if have_label:
            columns_to_keep.append(self.label_column)
        
        if mode == "train":
            columns_to_keep.extend(self.extra_columns_to_keep_train) # this is a list

        selected_data = data[columns_to_keep]

        # if have_label and (mode == "val" or mode == "test"):
        log_info(logger, f"Santitising labels of {path} file.\n")
        selected_data = self._label_fix(selected_data)
        if add_noise:
            log_info(logger, f"Flipping labels of {path} file.\n")
            selected_data = self._flip_labels(selected_data)
        
        if selected_data.isna().any().any(): 
            log_info(logger, f"Columns {selected_data.columns[selected_data.isna().any()].tolist()} have {selected_data.isna().sum().sum()} NaN values in total.")
            selected_data = selected_data.dropna() # drop NaN values; this could be NaN if the essay or label is None, so we drop the whole row
            log_info(logger, f"Removed rows with any NaN values. {len(selected_data)} samples remaining.\n")

        assert selected_data.isna().any().any() == False, "There are still NaN values in the data."
        assert selected_data.isnull().any().any() == False, "The are still null values in the data"

        return selected_data

    def _tokeniser_fn(self, sentence):
        if len(self.feature_to_tokenise) == 1: # only one feature
            return self.tokeniser(
                sentence[self.feature_to_tokenise[0]],
                truncation=True,
                max_length=self.max_length
            )
        # otherwise tokenise a pair of sentence
        return self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

    def get_hf_data(self, data_path_list, have_label, mode, add_noise):
        # we may combine the data from different versions
        for data_path in data_path_list:
            data = self._raw_to_processed(data_path, have_label, mode, add_noise)
            if 'all_data' in locals():
                all_data = pd.concat([all_data, data])
            else:
                all_data = data

        log_info(logger, f"Total number of {mode} samples: {len(all_data)}\n")
        assert all_data.isna().any().any() == False, "There are still NaN values in the data." # may occur due to the concat
        assert all_data.isnull().any().any() == False, "The are still null values in the data"

        # all_data.to_csv(f"tmp/all_{mode}_data.tsv", sep='\t', index=False) # save the data for debugging

        # if mode == "train":
        #     # add sample_id column
        #     all_data['sample_id'] = range(len(all_data))

        if self.llm_column in all_data.columns:
            all_data = all_data.drop(columns=[self.llm_column])   

        all_data_hf = Dataset.from_pandas(all_data, preserve_index=False) # convert to huggingface dataset
        
        # tokenise
        all_data_hf = all_data_hf.map(
            self._tokeniser_fn, 
            batched=True,
            remove_columns=self.feature_to_tokenise
        )
        if have_label:
            all_data_hf = all_data_hf.rename_column(self.label_column, 'labels')
        all_data_hf.set_format('torch')
        
        return all_data_hf
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def _get_dl(self, data_path_list, have_label, shuffle, mode, batch_size, add_noise: bool):
        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.seed)

        hf_data = self.get_hf_data(data_path_list=data_path_list, have_label=have_label, mode=mode, add_noise=add_noise)
        return DataLoader(
            hf_data,
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )
    def get_train_dl(self, data_path_list: list, batch_size: int):
        return self._get_dl(data_path_list, have_label=True, shuffle=True, mode="train", batch_size=batch_size, add_noise=True)
    
    def get_val_dl(self, data_path_list:list, batch_size: int):
        # depending on data_name, the labels can be in different file
        return self._get_dl(data_path_list, have_label=True, shuffle=False, mode="val", batch_size=batch_size, add_noise=False)
    
    def get_test_dl(self, data_path_list: list, batch_size: int = 32, have_label: bool = False, add_noise: bool = False):
        return self._get_dl(data_path_list, have_label=have_label, shuffle=False, mode="test", batch_size=batch_size, add_noise=add_noise) # we have labels in 2024 data
    