import random
import numpy as np
import pandas as pd
import os

import torch
import importlib
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import logging

from utils import log_info, log_debug, read_newsemp_file

logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

class NewsEmpPreprocessorFromRaw:
    """
    Preprocess the raw data to the format that can be used in other data processing pipelines here.
    It does the minimum processing required for the data.
    """
    def __init__(
        self,
        delta: float | None
    ):
        self.delta = delta
        
        # keeping them constant for now, can make them arguments if required
        self.label_shift = 3.0
        self.noise_level = 0.2
        self.label_min = 1.0
        self.label_max = 7.0
        self.label_column = "empathy"
        self.llm_column = "llm_empathy"
        self.columns_to_keep = ["essay", "article", self.label_column, self.llm_column, "article_id"]

    def _raw_to_processed(
        self, path: str, sanitise_labels: bool, add_noise: bool
    ) -> pd.DataFrame:
    
        data = read_newsemp_file(path)
        log_info(logger, f"Read {len(data)} samples from {path}")
        
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

        selected_data = data[[col for col in self.columns_to_keep if col in data.columns]] # keep only the columns that are in the data

        if sanitise_labels:
            log_info(logger, f"Santitising labels of {path} file.\n")
            selected_data = self._label_fix(selected_data)

        if add_noise:
            log_info(logger, f"Flipping labels of {path} file.\n")
            selected_data = self._flip_labels(selected_data)
        
        if selected_data.isna().any().any(): 
            log_info(logger, f"Columns {selected_data.columns[selected_data.isna().any()].tolist()} have {selected_data.isna().sum().sum()} NaN values in total.")
            selected_data.dropna(inplace=True) # drop NaN values; this could be NaN if the essay or label is None, so we drop the whole row
            log_info(logger, f"Removed rows with any NaN values. {len(selected_data)} samples remaining.\n")

        assert not selected_data.isna().any().any(), "There are still NaN values in the data."
        assert not selected_data.isnull().any().any(), "The are still null values in the data"

        return selected_data
    
    def _label_fix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Only keep the samples with the absolute difference between 'empathy' and 'llm_empathy' less than self.delta
        """
        assert self.label_column in data.columns, f"{self.label_column} column not found in the data"
        assert self.llm_column in data.columns, f"{self.llm_column} column not found in the data"
        
        if self.delta is not None:
            # Calculate the absolute difference between 'empathy' and 'llm_empathy'
            condition = np.abs(data[self.label_column] - data[self.llm_column]) < self.delta
            data = data[condition]
        
        data[self.label_column] = data[[self.label_column, self.llm_column]].mean(axis=1)

        data = data.drop(columns=[self.llm_column])

        return data
    
    def _flip_labels(self, data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        np.random.seed(seed)
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
    
    def process_data(self, data_paths: list[str], sanitise_labels: bool, add_noise: bool):
        # we may combine the data from different versions
        all_data = pd.DataFrame()
        for data_path in data_paths:
            data = self._raw_to_processed(
                path=data_path,
                sanitise_labels=sanitise_labels,
                add_noise=add_noise
            )
            all_data = pd.concat([all_data, data], ignore_index=True) if not all_data.empty else data

        log_info(logger, f"Total number of samples: {len(all_data)}\n")
        assert not all_data.isna().any().any(), "There are still NaN values in the data." # may occur due to the concat
        assert not all_data.isnull().any().any(), "The are still null values in the data"

        if self.llm_column in all_data.columns:
            all_data.drop(columns=[self.llm_column], inplace=True)

        all_data.rename(
            columns={
                self.label_column: "labels", 
                "essay": "text_1"
            },
            inplace=True
        )

        # add article information
        article = pd.read_csv("data/article-summarised.csv", index_col=0)
        all_data = pd.merge(all_data, article, on="article_id", how="left")
        all_data.drop(columns=["article_id", "text"], inplace=True)
        all_data.rename(columns={"summary_text": "text_2"}, inplace=True)

        return all_data

class BiEncoderDataCollator:
    def __init__(self, tokeniser_1, tokeniser_2):
        self.collator_1 = DataCollatorWithPadding(tokenizer=tokeniser_1)
        self.collator_2 = DataCollatorWithPadding(tokenizer=tokeniser_2)

    def __call__(self, batch):
        labels = [example["labels"] for example in batch] if "labels" in batch[0] else None

        batch_1 = [{k.replace("_1", ""): v for k, v in example.items() if "_1" in k} for example in batch]
        batch_2 = [{k.replace("_2", ""): v for k, v in example.items() if "_2" in k} for example in batch]

        batch_1 = self.collator_1(batch_1)
        batch_2 = self.collator_2(batch_2)

        final_batch = {
            "input_ids_1": batch_1["input_ids"],
            "attention_mask_1": batch_1["attention_mask"],
            "input_ids_2": batch_2["input_ids"],
            "attention_mask_2": batch_2["attention_mask"],
        }

        if labels is not None:
            final_batch["labels"] = torch.tensor(labels)

        return final_batch

class PairedTextDataModule:
    def __init__(
            self, 
            delta: float, 
            tokeniser_plms: list[str],
            is_separate_tokeniser: bool
        ):

        self.delta = delta
        self.is_separate_tokeniser = is_separate_tokeniser

        # keeping them constant for now, can make them arguments if required
        self.feature_to_tokenise = ["text_1", "text_2"]
        self.max_length = 512
        self.num_workers = 12

        self.tokeniser_plms = tokeniser_plms

        self.tokeniser = AutoTokenizer.from_pretrained(
            self.tokeniser_plms[0],
            use_fast=True,
            add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )

        if self.is_separate_tokeniser:
            self.tokeniser_extra = AutoTokenizer.from_pretrained(
                self.tokeniser_plms[1],
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
            )

            self.data_collator = BiEncoderDataCollator(tokeniser_1=self.tokeniser, tokeniser_2=self.tokeniser_extra)

        else:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)

    def _tokeniser_bi_encoder(self, sentence):
        tokenised_1 = self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            truncation=True,
            max_length=self.max_length
        )

        tokenised_2 = self.tokeniser_extra(
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

        return {
            'input_ids_1': tokenised_1['input_ids'],
            'attention_mask_1': tokenised_1['attention_mask'],
            'input_ids_2': tokenised_2['input_ids'],
            'attention_mask_2': tokenised_2['attention_mask']
        }
        
    def _tokeniser_cross_encoder(self, sentence):
        return self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

    def get_hf_data(
            self, data_paths: list[str], sanitise_newsemp_labels: bool, 
            add_noise: bool, is_newsemp: bool = True, do_augment: bool = False
        ):
        if do_augment:
            from textattack.augmentation import (
                # CLAREAugmenter, # requires tensorflow, which causing some issues with rocm GPU
                # BackTranslationAugmenter,
                # EmbeddingAugmenter, # time consuming, 3130 samples would take 24+ hours
                # EasyDataAugmenter # consist of four augmenters
                WordNetAugmenter
            )
        def _augment_and_combine(data: pd.DataFrame) -> pd.DataFrame:
            augmenters = [
                # CLAREAugmenter()
                # EmbeddingAugmenter(
                #     pct_words_to_swap=0.25,
                #     transformations_per_example=2
                # )
                WordNetAugmenter(
                    pct_words_to_swap=0.1,
                    transformations_per_example=1
                )

            ]

            augm_data_list = []
            for augmenter in augmenters:
                augm_text_1 = augmenter.augment_many(data["text_1"].tolist(), show_progress=True)
                augm_text_2 = augmenter.augment_many(data["text_2"].tolist(), show_progress=True)
                # each of the above returns list[list[str]], so we need to flatten them
                for each_text_1_variants, each_text_2_variants, label in zip(augm_text_1, augm_text_2, data["labels"].tolist()):
                    for text_1, text_2 in zip(each_text_1_variants, each_text_2_variants):
                        augm_data_list.append((text_1, text_2, label))

            # augm_data_list is list[tuple[str, str, float]] with a len of transformations_per_example * len(data)
            data_augms = pd.DataFrame(augm_data_list, columns=["text_1", "text_2", "labels"])
            data = pd.concat([data, data_augms], ignore_index=True)
            return data
        
        if is_newsemp:
            save_as = "data/newsemp_train_augmented.tsv" if do_augment else "data/newsemp_train.tsv"
        else:
            save_as = "data/empstories_train_augmented.tsv" if do_augment else "data/empstories_train.tsv"

        if os.path.exists(save_as):
            log_info(logger, f"Reading data from {save_as}")
            all_data = pd.read_csv(save_as, sep='\t')
            log_info(logger, f"Read {len(all_data)} samples from {save_as}")
        else:
            log_info(logger, f"Data not found at {save_as}. Processing from scratch.")
            if is_newsemp:
                newsemp_preprocessor = NewsEmpPreprocessorFromRaw(delta=self.delta)
                all_data = newsemp_preprocessor.process_data(
                    data_paths=data_paths,
                    sanitise_labels=sanitise_newsemp_labels,
                    add_noise=add_noise
                )
            else:
                # doesn't require much processing, so done here
                all_data = pd.DataFrame()
                for data_path in data_paths:
                    data = pd.read_csv(data_path)
                    log_info(logger, f"Read {len(data)} samples from {data_path}")
                    all_data = pd.concat([all_data, data], ignore_index=True) if not all_data.empty else data
                all_data["story_A"] = all_data["story_A"].str.replace("\n", "", regex=False)
                all_data["story_B"] = all_data["story_B"].str.replace("\n", "", regex=False)
                all_data.rename(
                    columns={
                        "story_A": "text_1",
                        "story_B": "text_2",
                        "similarity_empathy_human_AGG": "labels"
                    },
                    inplace=True
                )
                all_data = all_data[["text_1", "text_2", "labels"]]
                log_info(logger, f"Total number of samples: {len(all_data)}\n")

            if do_augment:
                all_data = _augment_and_combine(all_data)

            all_data.to_csv(save_as, sep='\t', index=False)
            log_info(logger, f"Saved the data to {save_as}")

        all_data_hf = Dataset.from_pandas(all_data, preserve_index=False) # convert to huggingface dataset
        
        # tokenise
        all_data_hf = all_data_hf.map(
            self._tokeniser_bi_encoder if self.is_separate_tokeniser else self._tokeniser_cross_encoder,
            batched=True,
            remove_columns=self.feature_to_tokenise
        )

        all_data_hf.set_format('torch')
        
        return all_data_hf
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def _get_dl(
            self, data_paths: list[str], shuffle: bool, batch_size: int, 
            sanitise_newsemp_labels: bool, add_noise: bool, seed: int | None = None,
            is_newsemp: bool = True, do_augment: bool = False
        ):

        if shuffle:
            # making sure the shuffling is reproducible
            g = torch.Generator()
            g.manual_seed(seed)

        hf_data = self.get_hf_data(
            data_paths=data_paths,
            sanitise_newsemp_labels=sanitise_newsemp_labels,
            add_noise=add_noise,
            is_newsemp=is_newsemp,
            do_augment=do_augment
        )

        return DataLoader(
            hf_data,
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g if shuffle else None
        )
    
    def get_train_dl(
            self, data_path_list: list, batch_size: int,
            sanitise_newsemp_labels: bool = True, add_noise: bool = False, seed: int | None = None,
            is_newsemp: bool = True
        ):
        return self._get_dl(
            data_path_list, shuffle=True, 
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise, seed=seed,
            is_newsemp=is_newsemp, do_augment=True
        )
    
    def get_val_dl(
            self, data_path_list:list, batch_size: int, 
            sanitise_newsemp_labels: bool = True, add_noise: bool = False,
            is_newsemp: bool = True
        ):
        # depending on data_name, the labels can be in different file
        return self._get_dl(
            data_path_list, shuffle=False, 
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise,
            is_newsemp=is_newsemp
        )
    
    def get_test_dl(
            self, data_path_list: list, batch_size: int = 32,
            sanitise_newsemp_labels: bool = True, add_noise: bool = False,
            is_newsemp: bool = True
        ):
        return self._get_dl(
            data_path_list, shuffle=False,
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise,
            is_newsemp=is_newsemp
        ) # we have labels in 2024 data


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
        
    
    def _raw_to_processed(self, path: str, have_label: bool, mode: str, sanitise_labels: bool, add_noise: bool) -> pd.DataFrame:
        log_info(logger, f"\nReading data from {path}")
        data = read_newsemp_file(path)
        
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
        if sanitise_labels:
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

    def get_hf_data(self, data_path_list, have_label, mode, sanitise_labels, add_noise):
        # we may combine the data from different versions
        for data_path in data_path_list:
            data = self._raw_to_processed(
                data_path, have_label, mode,
                sanitise_labels=sanitise_labels,
                add_noise=add_noise
            )
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
    
    def _get_dl(self, data_path_list, have_label, shuffle, mode, batch_size, sanitise_labels: bool, add_noise: bool):
        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.seed)

        hf_data = self.get_hf_data(
            data_path_list=data_path_list, have_label=have_label, mode=mode,
            sanitise_labels=sanitise_labels, add_noise=add_noise
        )

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
    
    def get_train_dl(
            self, data_path_list: list, batch_size: int,
            sanitise_labels: bool = True, add_noise: bool = True
        ):
        return self._get_dl(
            data_path_list, have_label=True, shuffle=True, mode="train", 
            batch_size=batch_size, sanitise_labels=sanitise_labels, add_noise=add_noise
        )
    
    def get_val_dl(
            self, data_path_list:list, batch_size: int, 
            sanitise_labels: bool = True, add_noise: bool = False
        ):
        # depending on data_name, the labels can be in different file
        return self._get_dl(
            data_path_list, have_label=True, shuffle=False, mode="val", 
            batch_size=batch_size, sanitise_labels=sanitise_labels, add_noise=add_noise
        )
    
    def get_test_dl(
            self, data_path_list: list, batch_size: int = 32, have_label: bool = False,
            sanitise_labels: bool = True, add_noise: bool = False
        ):
        return self._get_dl(
            data_path_list, have_label=have_label, shuffle=False, mode="test", 
            batch_size=batch_size, sanitise_labels=sanitise_labels, add_noise=add_noise
        ) # we have labels in 2024 data


