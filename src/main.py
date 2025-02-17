import os
import logging
import argparse
import transformers
import datetime
from omegaconf import OmegaConf

from utils import retrieve_newsemp_file_names, log_info

from paired_texts_modelling import PairedTextModelController
from ssl_modelling import SSLModelController

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    config = OmegaConf.load("config/config.yaml")
    
    # things coming from the config
    seeds = config.seeds
    num_epochs = config.num_epochs
    lr = config.lr
    train_bsz = config.train_bsz
    eval_bsz = config.eval_bsz
    delta = config.delta
    n_trials = config.n_trials
    parent_log_dir = config.parent_log_dir
    error_decay_factor = config.error_decay_factor
    loss_weight = config.loss_weight
    do_tune = config.do_tune
    do_train = config.do_train
    overwrite_parent_dir = config.overwrite_parent_dir
    approach = config.approach
    expt_name_postfix = config.expt_name_postfix
    debug = args.debug

    is_ssl = config.is_ssl
    unlbl_data_files = [
        "data/EmpathicStories/PAIRS (train).csv", 
        "data/EmpathicStories/PAIRS (dev).csv", 
        "data/EmpathicStories/PAIRS (test).csv"
    ]

    if not do_tune and not do_train and (overwrite_parent_dir is None):
        raise ValueError("Assuming you want to test only, please provide the overwrite_log_dir")

    expt_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{approach}_ssl_{is_ssl}'
    if expt_name_postfix is not None:
        expt_name += f"-{expt_name_postfix}"

    train_file_list, val_file_list, test_file_list = retrieve_newsemp_file_names(config)

    if overwrite_parent_dir is not None:
        log_info(logger, f"Using overwrite_logging_dir {overwrite_parent_dir}")
        log_info(logger, "MAKE SURE you DELETE the last directory manually which was not trained for all epochs.")
        parent_log_dir = os.path.normpath(overwrite_parent_dir) # normpath to remove trailing slashes if any
        expt_name = os.path.basename(parent_log_dir) # we need this for resuming Optuna
    else:
        parent_log_dir = os.path.join(parent_log_dir, expt_name)

    if debug:
        log_info(logger, "Debug mode")
        logger.setLevel(logging.DEBUG)
        seeds = seeds[:2] # reduce the number of seeds for debugging
        num_epochs = 2
        n_trials = 2
        parent_log_dir = "./log/debug"
        expt_name = "debug"
        if os.path.exists(parent_log_dir):
            os.system(f"rm -rf {parent_log_dir}")
        os.makedirs(parent_log_dir, exist_ok=True)

    if is_ssl:
        modelling = SSLModelController(
            train_file=train_file_list,
            val_file=val_file_list,
            test_file=test_file_list,
            lr=lr,
            train_bsz=train_bsz,
            eval_bsz=eval_bsz,
            num_epochs=num_epochs,
            delta=delta,
            expt_name=expt_name,
            debug=debug,
            do_tune=do_tune,
            do_train=do_train,
            do_test=True, # automatically, not done during hyperparameter tuning
            error_decay_factor=error_decay_factor,
            loss_weight=loss_weight,
            approach=approach,
            unlbl_data_files=unlbl_data_files
        )
    else:
        modelling = PairedTextModelController(
            train_file=train_file_list,
            val_file=val_file_list,
            test_file=test_file_list,
            lr=lr,
            train_bsz=train_bsz,
            eval_bsz=eval_bsz,
            num_epochs=num_epochs,
            delta=delta,
            expt_name=expt_name,
            debug=debug,
            do_tune=do_tune,
            do_train=do_train,
            do_test=True, # automatically, not done during hyperparameter tuning
            error_decay_factor=error_decay_factor,
            loss_weight=loss_weight,
            approach=approach
        )

    modelling.tune_train_test(
        n_trials=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )
