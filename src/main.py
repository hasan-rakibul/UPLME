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
    main_data = config.main_data

    is_ssl = config.is_ssl

    newsemp_train_files, newsemp_val_files, newsemp_test_files = retrieve_newsemp_file_names(config)
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
        raise ValueError("Assuming you want to test only, please provide the overwrite_log_dir")

    expt_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{main_data}_{approach}_ssl-{is_ssl}'
    if expt_name_postfix is not None:
        expt_name += f"-{expt_name_postfix}"

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
            labelled_train_files=labelled_train_files,
            val_files=val_files,
            test_files=test_files,
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
            main_data=main_data,
            unlbl_data_files=unlbl_data_files
        )
    else:
        modelling = PairedTextModelController(
            labelled_train_files=labelled_train_files,
            val_files=val_files,
            test_files=test_files,
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
            main_data=main_data
        )

    modelling.tune_train_test(
        n_trials=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )
