import os
import logging
import argparse
import glob
import transformers
import datetime
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation
from functools import partial

from utils import retrieve_file_names, log_info, resolve_logging_dir, process_seedwise_metrics, DelayedStartEarlyStopping, resolve_num_steps
from preprocess import BiEncoderDataModule
from bi_encoder_model import LitBiEncoder

logger = logging.getLogger(__name__)

class BiEncoderModelling(object):
    def __init__(
        self,
        train_file: list[str],
        val_file: list[str],
        test_file: list[str],
        lr: float,
        train_bsz: int,
        eval_bsz: int,
        num_epochs: int,
        delta: float,
        expt_name: str,
        debug: bool,
    ):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
         
        self.lr = lr
        self.train_bsz = train_bsz
        self.eval_bsz = eval_bsz
        self.num_epochs = num_epochs

        self.delta = delta
        self.expt_name = expt_name
        self.debug = debug
        self.error_decay_factor = 0.5
        self.loss_weights = [0]

        self.enable_early_stopping = True
        self.early_stopping_start_epoch = 5
        self.enable_checkpointing = True
        self.plm_names = ["roberta-base", "roberta-base"]
        self.save_uc_metrics = False
        self.warmup_ratio = 0.06

        self.dm = BiEncoderDataModule(
            delta=self.delta,
            tokeniser_plms=self.plm_names,
        )
           
    def _prepare_trainer(self, curr_log_dir: str, extra_callbacks: list | None = None):
        callbacks = []

        if self.enable_early_stopping:
            early_stopping = DelayedStartEarlyStopping(
                start_epoch=self.early_stopping_start_epoch,
                monitor="val_ccc",
                patience=2,
                mode="max",
                min_delta=0,
                verbose=True
            )
            callbacks.append(early_stopping)
        else:
            log_info(logger, "Early stopping disabled")

        if self.enable_checkpointing:            
            checkpoint = ModelCheckpoint(
                save_top_k=1 # saves the last checkpoint; no need to save_last=True as it will save another checkpoint unnecessarily
            )
            
            callbacks.append(checkpoint)
            
        callbacks.extend(extra_callbacks) if extra_callbacks else None

        wandb_logger = WandbLogger(
            name=self.expt_name,
            project="NoisEmpathy",
            save_dir=curr_log_dir,
            offline=self.debug
        )

        trainer = L.Trainer(
            max_epochs=self.num_epochs,
            default_root_dir=curr_log_dir,
            deterministic=True,
            logger=wandb_logger,
            log_every_n_steps=10,
            callbacks=callbacks,
            devices="auto",
            enable_checkpointing=self.enable_checkpointing,
            limit_train_batches=0.1 if self.debug else 1.0 
        )

        return trainer

    def _seed_wise_train_validate(self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None):
        L.seed_everything(seed)

        train_dl = self.dm.get_train_dl(
            data_path_list=self.train_file,
            batch_size=self.train_bsz,
            sanitise_labels=True,
            add_noise=False,
            seed=seed
        )

        val_dl = self.dm.get_val_dl(
            data_path_list=self.val_file,
            batch_size=self.eval_bsz,
            sanitise_labels=True,
            add_noise=False
        )

        trainer = self._prepare_trainer(curr_log_dir=curr_log_dir, extra_callbacks=extra_callbacks)

        num_training_steps = len(train_dl) * self.num_epochs
        num_warmup_steps = int(self.warmup_ratio * len(train_dl) * 10) # 10 epochs of warmup calculation, like the RoBERTa paper

        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        if os.path.exists(curr_log_dir):
            log_info(logger, f"Seed-level logging directory already exists: {curr_log_dir}. So, validating on the saved ckpt...")
            ckpt_list = glob.glob(os.path.join(curr_log_dir, "**/*.ckpt"), recursive=True)
            assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
            best_model_ckpt = ckpt_list[0]
        else:
            log_info(logger, f"Training from scratch")  
            with trainer.init_module():
                # model created here directly goes to GPU
                model = LitBiEncoder(
                    plm_names=self.plm_names,
                    lr=self.lr,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=num_warmup_steps,
                    log_dir=curr_log_dir,
                    save_uc_metrics=self.save_uc_metrics,
                    error_decay_factor=self.error_decay_factor,
                    loss_weights=self.loss_weights
                )
            
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
            )

            # getting the best model from the trainer
            best_model_ckpt = trainer.checkpoint_callback.best_model_path
        
        # final validation at the end of training
        log_info(logger, f"Loading the best model from {best_model_ckpt}")
        with trainer.init_module(empty_init=True):
            model = LitBiEncoder.load_from_checkpoint(best_model_ckpt)

        trainer.validate(model=model, dataloaders=val_dl)

        metrics = {
            "val_pcc": trainer.callback_metrics["val_pcc"].item(),
            "val_ccc": trainer.callback_metrics["val_ccc"].item(),
            "val_rmse": trainer.callback_metrics["val_rmse"].item()
        }

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

        return best_model_ckpt, metrics
    
    def evaluate(self, model_path: str):
        tester = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        test_dl = self.dm.get_test_dl(
            data_path_list=self.test_file, batch_size=16, have_label=True,
            sanitise_labels=False, add_noise=False
        )

        with tester.init_module(empty_init=True):
            model = LitBiEncoder.load_from_checkpoint(model_path, save_uc_metrics=self.save_uc_metrics)

        tester.test(model=model, dataloaders=test_dl, verbose=True)

        try:
            metrics = {
                "test_pcc": tester.callback_metrics["test_pcc"].item(),
                "test_ccc": tester.callback_metrics["test_ccc"].item(),
                "test_rmse": tester.callback_metrics["test_rmse"].item()
            }
        except KeyError:
            metrics = {}
        
        return metrics
    
    def train_test(
        self,
        do_test: bool,
        seeds: list[int],
        parent_log_dir: str
    ) -> None:

        results = []
        for seed in seeds:
            log_info(logger, f"Current seed: {seed}")
            curr_log_dir = os.path.join(parent_log_dir, f"seed_{seed}")

            best_model_ckpt, metrics = self._seed_wise_train_validate(
                seed=seed,
                curr_log_dir=curr_log_dir
            )
            
            if do_test:
                # subsequent testing
                log_info(logger, f"Testing right after training from {best_model_ckpt}")
                test_metrics = self.evaluate(best_model_ckpt)
                metrics = {**metrics, **test_metrics} # merge the two dictionaries 

            metrics["seed"] = seed
            log_info(logger, f"Metrics: {metrics}")
            results.append(metrics)
        save_as = os.path.join(parent_log_dir, "results.csv")
        process_seedwise_metrics(results, save_as)

    def optuna_objective(self, trial: optuna.trial.Trial, optuna_seed: int, optuna_log_dir: str) -> float:
        self.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        self.train_bsz = trial.suggest_int("train_bsz", 8, 32, step=8)

        self.error_decay_factor = trial.suggest_float("error_decay_factor", 0.0, 3.0, step=0.5)
        penalty_weight = trial.suggest_float("penalty_weight", 0.0, 100.0)
        self.loss_weights = [penalty_weight]

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        _, metrics = self._seed_wise_train_validate(
            seed=optuna_seed,
            curr_log_dir=optuna_log_dir,
            extra_callbacks=[pruning_callback]
        )

        return metrics["val_ccc"]

    def tune_train_test(self, do_tune: bool, do_test: bool, n_trails: int, parent_log_dir: str, seeds: int = 0) -> None:
        if do_tune:
            optuna_log_dir = os.path.join(parent_log_dir, "optuna_logs")
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                study_name=self.expt_name,
                storage=f"sqlite:///{optuna_log_dir}/optuna.db",
                load_if_exists=True
            )

            optuna_seed = seeds[0] # using the first seed for optuna tuning
            
            objective_params = partial(
                self.optuna_objective,
                optuna_seed=optuna_seed,
                optuna_log_dir=optuna_log_dir
            )

            study.optimize(objective_params, n_trials=n_trails, show_progress_bar=False)

            study.trials_dataframe().to_csv(os.path.join(optuna_log_dir, "trial_results.csv"), index=False)
            
            best_trial = study.best_trial
            log_info(logger, f"Best trial:{best_trial.value}")

            with open(os.path.join(optuna_log_dir, "best_trial_params.txt"), 'w') as f:
                for key, value in best_trial.params.items():
                    f.write(f"{key}: {value}\n")
            
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_image(os.path.join(parent_log_dir, "Optuna_slice.pdf"))

            fig_imp = optuna.visualization.plot_param_importances(study)
            fig_imp.write_image(os.path.join(parent_log_dir, "Optuna_param_importances.pdf"))

            # update the parameters and retrain
            self.lr = best_trial.params["lr"]
            self.train_bsz = best_trial.params["train_bsz"]
            self.error_decay_factor = best_trial.params["error_decay_factor"]
            self.loss_weights = [best_trial.params["penalty_weight"]]
        
        self.train_test(do_test=do_test, seeds=seeds, parent_log_dir=parent_log_dir)


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

    expt_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.approach})'
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
        parent_log_dir = "./tmp"
        num_epochs = 2
        n_trials = 2

    bi_encoder = BiEncoderModelling(
        train_file=train_file_list,
        val_file=val_file_list,
        test_file=test_file_list,
        lr=lr,
        train_bsz=train_bsz,
        eval_bsz=eval_bsz,
        num_epochs=num_epochs,
        delta=delta,
        expt_name=expt_name,
        debug=args.debug
    )

    bi_encoder.tune_train_test(
        do_tune=args.tune_hparams,
        do_test=True,
        n_trails=n_trials,
        parent_log_dir=parent_log_dir,
        seeds=seeds
    )
