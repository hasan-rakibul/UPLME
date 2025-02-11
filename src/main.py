import os
import logging
import argparse
import transformers
import datetime
from omegaconf import OmegaConf

from utils import retrieve_file_names, log_info

from bi_encoder_modelling import PairedTextModelTrainer
from cross_encoder_modelling import CrossEncoderModelling

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-a", "--approach", type=str, default="single-prob", help="Approach: basic, single-prob, ensemble-prob")
    parser.add_argument("-e", "--expt_name_postfix", type=str, default="", help="Experiment name postfix")
    parser.add_argument("-o", "--overwrite_logging_dir", type=str, default=None, help="Overwrite logging directory")
    parser.add_argument("-t", "--tune_hparams", action="store_true", help="Tune hyperparameters")

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
    plm_names = config.plm_names
    error_decay_factor = config.error_decay_factor
    loss_weights = config.loss_weights

    expt_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.approach}'
    if args.expt_name_postfix != "":
        expt_name += f"-{args.expt_name_postfix}"

    train_file_list, val_file_list, test_file_list = retrieve_file_names(config)

    if args.overwrite_logging_dir is not None:
        log_info(logger, f"Using overwrite_logging_dir {args.overwrite_logging_dir}")
        log_info(logger, "MAKE SURE you DELETE the last directory manually which was not trained for all epochs.")
        parent_log_dir = args.overwrite_logging_dir
    else:
        parent_log_dir = os.path.join(parent_log_dir, expt_name)

    if args.debug:
        log_info(logger, "Debug mode")
        logger.setLevel(logging.DEBUG)
        seeds = seeds[:2] # reduce the number of seeds for debugging
        num_epochs = 2
        n_trials = 2
        parent_log_dir = "./log/debug"
        if os.path.exists(parent_log_dir):
            os.system(f"rm -rf {parent_log_dir}")
        os.makedirs(parent_log_dir, exist_ok=True)

    modelling = PairedTextModelTrainer(
        train_file=train_file_list,
        val_file=val_file_list,
        test_file=test_file_list,
        lr=lr,
        train_bsz=train_bsz,
        eval_bsz=eval_bsz,
        num_epochs=num_epochs,
        delta=delta,
        expt_name=expt_name,
        debug=args.debug,
        do_tune=args.tune_hparams,
        do_test=True,
        plm_names=plm_names,
        error_decay_factor=error_decay_factor,
        loss_weights=loss_weights
    )

    # modelling = CrossEncoderModelling(
    #     train_file=train_file_list,
    #     val_file=val_file_list,
    #     test_file=test_file_list,
    #     lr=lr,
    #     train_bsz=train_bsz,
    #     eval_bsz=eval_bsz,
    #     num_epochs=num_epochs,
    #     delta=delta,
    #     expt_name=expt_name,
    #     debug=args.debug,
    #     do_tune=args.tune_hparams,
    #     do_test=True
    # )

    modelling.tune_train_test(
        n_trials=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )
