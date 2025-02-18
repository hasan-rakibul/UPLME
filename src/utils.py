import os
import datetime
import logging
import numpy as np
import pandas as pd
import torch
import glob
from omegaconf import OmegaConf
import warnings
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# collected and slightly updated from https://github.com/Lightning-AI/pytorch-lightning/issues/16881#issuecomment-1447429542
class DelayedStartEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set start_epoch to None or 0 for no delay
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: "L.Trainer", l_module: "L.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_train_epoch_end(trainer, l_module)

    def on_validation_end(self, trainer: "L.Trainer", l_module: "L.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_validation_end(trainer, l_module)

def plot_uncertainy(data: dict, save_as: str) -> None:
    mean = data.get("mean", 0)
    unc = np.sqrt(data.get("var", 0))
    labels = data.get("labels", 0)
    # noise = data.get("noise", 0)

    _, ax = plt.subplots(1, 1)
    sc = ax.scatter(labels, mean, c=unc, cmap="coolwarm")
    # ax.errorbar(labels, mean, yerr=unc, fmt='o', alpha=0.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Uncertainty")

    plt.savefig(save_as)
    plt.close()
    log_info(logger, f"Saved plot at {save_as}")

def get_trainer(
        config, devices="auto", extra_callbacks=None, enable_checkpointing=True, enable_early_stopping=True,
        debug: bool = False, expt_name: str = None, logging_dir: str = None
    ):
    """
    By default, we have EarlyStopping.
    ModelCheckpoint is enabled if enable_checkpointing is True.
    If you want to add more callbacks, pass them in extra_callbacks.
    """
    callbacks = []

    if enable_early_stopping:
        early_stopping = DelayedStartEarlyStopping(
            start_epoch=config.early_stopping_start_epoch,
            monitor="val_ccc",
            patience=2,
            mode="max",
            min_delta=0,
            verbose=True
        )
        # maybe val_pcc is not a good idea as pcc is not correlated beween val and test
        # early_stopping = EarlyStopping(
        #     monitor="val_loss",
        #     patience=2,
        #     mode="min",
        #     min_delta=0.1
        # )
        callbacks.append(early_stopping)
    else:
        log_info(logger, "Early stopping disabled")

    if enable_checkpointing:
        # have a ModelCheckpoint callback
        # checkpoint = ModelCheckpoint(
        #     monitor="val_ccc",
        #     save_top_k=1,
        #     mode="max"
        # )
        # checkpoint = ModelCheckpoint(
        #     monitor="val_loss",
        #     save_top_k=1,
        #     mode="min"
        # )
        
        checkpoint = ModelCheckpoint(
            save_top_k=1 # saves the last checkpoint; no need to save_last=True as it will save another checkpoint unnecessarily
        )
        
        callbacks.append(checkpoint)
        
    callbacks.extend(extra_callbacks) if extra_callbacks else None

    wandb_logger = WandbLogger(
        name=expt_name,
        project="NoisEmpathy",
        save_dir=logging_dir
    )

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=config.logging_dir,
        deterministic=True,
        logger=wandb_logger if not debug else None,
        log_every_n_steps=10,
        callbacks=callbacks,
        devices=devices,
        enable_checkpointing=enable_checkpointing,
        limit_train_batches=0.1 if debug else 1.0 
    )

    return trainer

def read_newsemp_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t', na_values="unknown") # some column includes "unknown"
    elif file_path.endswith(".csv"):
        # 2024 raw data are in csv. The essay has commas, and placed inside double quotes
        # Further, tt has \" inside the quoted text, for example, "I am a \"good\" person"
        df = pd.read_csv(file_path, quotechar='"', escapechar="\\")

        # "2024" has different column names
        df = df.rename(columns={
            "person_essay": "essay",
            "person_empathy": "empathy"
        })
    else:
        raise ValueError(f"File extension not supported: {file_path}")
    return df

def resolve_logging_dir(config):
    if "resume_train_from_checkpoint" in config:
        assert os.path.exists(config.resume_train_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.resume_train_from_checkpoint.split("/")[:-4] # logs/.../ ; calculate from end as we may have ./logs/ or just logs/
        logging_dir = os.path.join(*path_list)   
    elif "finetune_from_checkpoint" in config:
        assert os.path.exists(config.finetune_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.finetune_from_checkpoint.split("/")[:-4]
        # add additional directory for finetuning
        path_list.append("finetune")
        logging_dir = os.path.join(*path_list)
    elif "test_from_checkpoint" in config:
        assert os.path.exists(config.test_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.test_from_checkpoint.split("/")[:-4]
        logging_dir = os.path.join(*path_list)
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
    return logging_dir

def resolve_seed_wise_checkpoint(parent_dir: str, seed: float) -> str:
    ckpt = glob.glob(os.path.join(parent_dir, f"**/seed_{seed}/**/*.ckpt"), recursive=True)
    assert len(ckpt) == 1, f"Found {len(ckpt)} checkpoints for seed {seed}"
    return ckpt[0]

def process_seedwise_metrics(results: list, save_as: str) -> None:
    results_df = pd.DataFrame(results)

    results_df.set_index("seed", inplace=True)
    results_df = results_df.round(3)
    
    # post-processing
    mean_row = results_df.mean(numeric_only=True).round(3)
    std_row = results_df.std(numeric_only=True).round(3)
    median_row = results_df.median(numeric_only=True).round(3)
    
    best_scores = pd.DataFrame(index=["best"], columns=results_df.columns)
    for col in results_df.columns:
        if col.endswith("_rmse"):
            best_scores.loc["best", col] = results_df[col].min()
        else:
            best_scores.loc["best", col] = results_df[col].max()
    best_row = best_scores.loc["best", :].round(3)
    
    # Assign a label to identify each row
    mean_row.name = "mean"
    std_row.name = "std"
    median_row.name = "median"

    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T, median_row.to_frame().T])
    
    results_df.to_csv(save_as, index=True)
    log_info(logger, f"Results saved at {save_as}")

    # print the result, in LaTeX-table style
    log_info(logger, " & ".join(results_df.columns))
    log_info(logger, "\nMedian(Best)")
    log_info(logger, " & ".join([f"${median}({best})$" for median, best in zip(median_row, best_row)]))

@rank_zero_only
def log_info(logger, msg):
    logger.info(msg)

@rank_zero_only
def log_debug(logger, msg):
    logger.debug(msg)

def retrieve_newsemp_file_names(config: OmegaConf) -> tuple[list, list, list]:
    llm = "llama"
    train_attr = f"train_{llm}"
    val_attr = f"val_{llm}"
    test_attr = f"test_{llm}"
    
    train_file_list = []
    for data in config.train_data:
        train_file_list.append(getattr(config[data], train_attr))
        if data != config.val_data:
            # we don't want to include val data in the training data for the same year
            train_file_list.append(getattr(config[data], val_attr))

    val_file_list = [getattr(config[config.val_data], val_attr)]
    test_file_list = [getattr(config[config.val_data], test_attr)]

    return train_file_list, val_file_list, test_file_list

def prepare_test_config(config: OmegaConf) -> OmegaConf:
    config.batch_size = config.eval_batch_size
    config.test_file_list = []
    config.make_ready_for_submission = False 
    config.have_label = True
    for data in config.test_data:
        config.test_file_list.append(getattr(config[data], config.test_split))
        if data in [2023, 2022]:
            # only way to test is through CodaLab submission
            config.make_ready_for_submission = True
            config.have_label = False

    return config

def resolve_num_steps(config: OmegaConf, train_dl) -> tuple:
    # number of training steps is required for linear scheduler
    num_training_steps = len(train_dl) * config.num_epochs
    if "num_warmup_steps" in config:
        num_warmup_steps = config.num_warmup_steps
        if num_warmup_steps > config.num_training_steps:
            raise ValueError("Number of warmup steps cannot be greater than the number of training steps.")
        if "warmup_ratio" in config:
            warnings.warn("Both num_warmup_steps and warmup_ratio are provided. Ignoring warmup_ratio.")
    else:
        num_warmup_steps = int(config.warmup_ratio * len(train_dl) * 10) # 10 epochs of warmup calculation, like the RoBERTa paper
    log_info(logger, f"Number of warmup steps: {num_warmup_steps}")

    return num_training_steps, num_warmup_steps
