import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualization

import os
import datetime
import logging
import argparse
import transformers
from omegaconf import OmegaConf
from functools import partial
import lightning as L

from utils import log_info, log_debug, get_trainer, prepare_train_config, resolve_num_steps
from preprocess import DataModuleFromRaw
from model import LitProbabilisticPLMEnsemble

logger = logging.getLogger(__name__)

def objective(
        trial: optuna.trial.Trial,
        config: OmegaConf,
        seed: int,
        lr: float,
        objectives: list,
        debug: bool,
        expt_name: str,
        logging_dir: str,
        batch_size: int,
        delta: float,
        plm_names: list
    ) -> float:

    # things to tune
    consistency_weight = trial.suggest_float("consistency_weight", 0.0, 100.0)
    # penalty_weight = trial.suggest_float("penalty_weight", 0.0, 100.0)
    penalty_weight = 0.0
    loss_weights = [consistency_weight, penalty_weight]
    
    L.seed_everything(seed)

    datamodule = DataModuleFromRaw(delta=delta, seed=seed)
    
    train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list, batch_size=batch_size)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list, batch_size=batch_size)

    if config.lr_scheduler_type == "linear":
        num_training_steps, num_warmup_steps = resolve_num_steps(config, train_dl)
    
    if len(objectives) > 1:
        # multi objective optimisation
        extra_callbacks = None
    else:
        extra_callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        ]

    trainer = get_trainer(
        config,
        extra_callbacks=extra_callbacks,
        enable_checkpointing=True,
        debug=debug,
        expt_name=expt_name,
        logging_dir=logging_dir
    )

    with trainer.init_module():
        model = LitProbabilisticPLMEnsemble(
            plm_names=plm_names,
            lr=lr,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            loss_weights=loss_weights
        )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    log_debug(logger, f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    # log_debug(logger, f"Best model score: {trainer.callback_metrics["val_pcc"]}")

    best_model_ckpt = trainer.checkpoint_callback.best_model_path
    with trainer.init_module(empty_init=True):
        model = LitProbabilisticPLMEnsemble.load_from_checkpoint(best_model_ckpt)
    trainer.validate(model=model, dataloaders=val_dl)

    log_debug(logger, f"Best model validation score: {trainer.callback_metrics['val_pcc']}")

    metrics = []
    for objective in objectives:
        metrics.append(trainer.callback_metrics[objective].item())
    
    return tuple(metrics)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-r", "--resume_dir", type=str, default=None, help="Resume from a previous optuna run")
    parser.add_argument("-n", "--n_trials", type=int, default=100, help="Number of optuna trails")
    parser.add_argument("-e", "--expt_name", type=str, default="tune", help="Experiment name")

    # variables could be passed as arguments but for now, we are hardcoding them
    approach = "ensemble-probabilistic"
    objectives = ["val_ccc"]
    directions = ["maximize"]
    seed = 0

    assert len(objectives) == len(directions), "Number of objectives and directions must match"

    args = parser.parse_args()

    transformers.logging.set_verbosity_error()

    config_hparam = OmegaConf.load("config/config_train.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_hparam)

    config.expt_name_postfix = args.expt_name
    config = prepare_train_config(config, approach=approach)

    if args.resume_dir is not None:
        storage = f"sqlite:///{args.resume_dir}/optuna.db"
        log_info(logger, f"Resuming from {args.resume_dir}")
        config.logging_dir = args.resume_dir
    else:
        if args.debug:
            logger.setLevel(logging.DEBUG)
            config.logging_dir = "/tmp"
            log_info(logger, f"Running in debug mode. Logging to {config.logging_dir}")
            args.n_trials = 1

        config.logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
        os.makedirs(config.logging_dir, exist_ok=True)
        storage = f"sqlite:///{config.logging_dir}/optuna.db"

    # atm, multi objective optimisation does not support pruning
    # but we have early stopping in the trainer, so it's fine
    pruner = optuna.pruners.NopPruner() if len(objectives) > 1 else optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=config.expt_name,
        storage=storage,
        directions=directions,
        pruner=pruner,
        load_if_exists=True
    )

    study.set_metric_names(objectives) if isinstance(objectives, list) else study.set_metric_names(list(objectives)) # requires for ListConfig

    objective_param = partial(
        objective,
        config=config,
        seed=seed,
        lr=config.lrs[0],
        objectives=objectives,
        debug=args.debug,
        expt_name=config.expt_name,
        logging_dir=config.logging_dir,
        batch_size=config.batch_sizes[0],
        delta=config.delta,
        plm_names=config.plm_names
    )
    study.optimize(objective_param, n_trials=args.n_trials, show_progress_bar=False)

    trial_results = study.trials_dataframe()
    trial_results.to_csv(os.path.join(config.logging_dir, "trials_results.csv"))

    log_info(logger, f"Number of finished trials: {len(study.trials)}")
    if len(objectives) > 1:
        best_trials = study.best_trials
        with open(os.path.join(config.logging_dir, "best_trials_params.txt"), 'w') as f:
            for i, best_trail in enumerate(best_trials):
                f.write(f"Best trial {i}:\n")
                for key, value in best_trail.params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
    else:
        best_trial = study.best_trial
        log_info(logger, f"Best trial:{best_trial.value}")

        with open(os.path.join(config.logging_dir, "best_trial_params.txt"), 'w') as f:
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # only for single objective optimisation
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_image(os.path.join(config.logging_dir, "Optuna_slice.pdf"))

    fig_imp = optuna.visualization.plot_param_importances(study)
    fig_imp.write_image(os.path.join(config.logging_dir, "Optuna_param_importances.pdf"))
