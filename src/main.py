import os
import logging
import transformers

import hydra
from omegaconf import DictConfig

from utils import retrieve_newsemp_file_names, log_info

from paired_texts_modelling import PairedTextModelController
from ssl_modelling import SSLModelController

logger = logging.getLogger(__name__)

import torch
# MI250X GPU has Tensor core, so recommeded to use high or medium precision
# as opposed to highest precision (default) for faster computation 
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    transformers.logging.set_verbosity_error()
    logger.info(f"Experiment name: {cfg.expt}")
    logger.info(f"Config: {cfg}")

    parent_log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    expt_name = hydra.core.hydra_config.HydraConfig.get().job.name
    
    # things coming from the config
    seeds = cfg.seeds
    # num_epochs = cfg.num_epochs
    lr = cfg.lr
    train_bsz = cfg.train_bsz
    eval_bsz = cfg.eval_bsz
    delta = cfg.delta
    n_trials = cfg.n_trials
    error_decay_factor = cfg.error_decay_factor
    do_tune = cfg.do_tune
    do_train = cfg.do_train
    overwrite_parent_dir = cfg.expt.overwrite_parent_dir
    approach = cfg.approach
    main_data = cfg.main_data

    is_ssl = cfg.is_ssl

    newsemp_train_files, newsemp_val_files, newsemp_test_files = retrieve_newsemp_file_names(cfg)
    empstories_train_files = ["data/EmpathicStories/PAIRS (train).csv"]
    empstories_val_files = ["data/EmpathicStories/PAIRS (dev).csv"]
    empstories_test_files = ["data/EmpathicStories/PAIRS (test).csv"]
    if main_data == "newsemp":
        labelled_train_files = newsemp_train_files
        val_files = newsemp_val_files
        test_files = newsemp_test_files
        unlbl_data_files = empstories_train_files + empstories_val_files + empstories_test_files
    elif main_data == "empstories":
        labelled_train_files = empstories_train_files
        val_files = empstories_val_files
        test_files = empstories_test_files
        unlbl_data_files = newsemp_train_files + newsemp_val_files + newsemp_test_files
    else:
        raise ValueError("main_data should be either newsemp or empstories")
    log_info(logger, f"Train data: {labelled_train_files}\tVal data: {val_files}\tTest data: {test_files}")

    if not do_tune and not do_train and (overwrite_parent_dir is None):
        raise ValueError("Assuming you want to test only, please provide the overwrite_parent_dir")

    if overwrite_parent_dir is not None:
        log_info(logger, f"Using overwrite_logging_dir {overwrite_parent_dir}")
        assert os.path.isdir(overwrite_parent_dir), f"{overwrite_parent_dir} is not a directory \
            Note it must be a **parent** diretory like outputs/yyyy-mm-dd/hh-mm-ss_xx."
        log_info(logger, "MAKE SURE you DELETE the last directory manually which was not trained for all epochs.")
        parent_log_dir = os.path.normpath(overwrite_parent_dir) # normpath to remove trailing slashes if any
        expt_name = os.path.basename(parent_log_dir) # we need this for resuming Optuna

    debug = False
    if expt_name.startswith("debug"):
        debug = True
        log_info(logger, "Debug mode")
        logger.setLevel(logging.DEBUG)
        seeds = seeds[:2] # reduce the number of seeds for debugging
        # num_epochs = 2
        cfg.max_steps = 50
        cfg.val_check_interval = 5
        n_trials = 2

    if is_ssl:
        modelling = SSLModelController(
            labelled_train_files=labelled_train_files,
            val_files=val_files,
            test_files=test_files,
            lr=lr,
            train_bsz=train_bsz,
            eval_bsz=eval_bsz,
            # num_epochs=num_epochs,
            max_steps=cfg.max_steps,
            val_check_interval=cfg.val_check_interval,
            delta=delta,
            expt_name=expt_name,
            debug=debug,
            do_tune=do_tune,
            do_train=do_train,
            do_test=True, # automatically, not done during hyperparameter tuning
            error_decay_factor=error_decay_factor,
            lambda_1=cfg.lambda_1,
            lambda_2=cfg.expt.lambda_2,
            lambda_3=cfg.expt.lambda_3,
            approach=approach,
            main_data=main_data,
            unlbl_data_files=unlbl_data_files,
            lbl_split=cfg.lbl_split
        )
    else:
        modelling = PairedTextModelController(
            labelled_train_files=labelled_train_files,
            val_files=val_files,
            test_files=test_files,
            lr=lr,
            train_bsz=train_bsz,
            eval_bsz=eval_bsz,
            # num_epochs=num_epochs,
            max_steps=cfg.max_steps,
            val_check_interval=cfg.val_check_interval,
            delta=delta,
            expt_name=expt_name,
            debug=debug,
            do_tune=do_tune,
            do_train=do_train,
            do_test=True, # automatically, not done during hyperparameter tuning
            error_decay_factor=error_decay_factor,
            lambda_1=cfg.lambda_1,
            approach=approach,
            main_data=main_data
        )

    modelling.tune_train_test(
        n_trials=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )

if __name__ == "__main__":
    main()
