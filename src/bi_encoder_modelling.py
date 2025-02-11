import torch
import torch.nn.functional as F
import lightning as L
from transformers import (
    AutoModel, 
    get_linear_schedule_with_warmup
)
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error
import logging
import numpy as np

import os
import glob

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation
from functools import partial

from utils import log_info, process_seedwise_metrics, DelayedStartEarlyStopping 
from preprocess import PairedTextDataModule

from utils import log_info

logger = logging.getLogger(__name__)

class BiEncoderModel(torch.nn.Module):
    def __init__(
        self,
        plm_names: list[str]
    ):
        super().__init__()
        plm_1, plm_2 = plm_names
        self.plm_1 = AutoModel.from_pretrained(plm_1)
        self.plm_2 = AutoModel.from_pretrained(plm_2)

        self.fc_concat = torch.nn.Linear(1536, 768)
        self.fc_m = torch.nn.Linear(768, 1)
        self.fc_v = torch.nn.Linear(768, 1)

        self.min_score = 1.0
        self.max_score = 7.0

    def forward(self, batch):
        output_1 = self.plm_1(
            input_ids=batch['input_ids_1'],
            attention_mask=batch['attention_mask_1']
        )
        output_2 = self.plm_2(
            input_ids=batch['input_ids_2'],
            attention_mask=batch['attention_mask_2']
        )

        x_1 = output_1.last_hidden_state[:, 0, :]
        x_2 = output_2.last_hidden_state[:, 0, :]

        x = torch.cat((x_1, x_2), dim=1)
        x = F.relu(self.fc_concat(x))
        x = F.dropout(x, p=0.25)

        unbounded_mean = self.fc_m(x)
        scaled_mean = self.min_score + (self.max_score - self.min_score) * torch.sigmoid(unbounded_mean)
        
        var = F.softplus(self.fc_v(x)) # variance must be positive

        return scaled_mean.squeeze(), var.squeeze()
    
class CrossEncoderModel(torch.nn.Module):
    def __init__(self, plm_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(plm_name)
        
        self.fc_m = torch.nn.Linear(1024, 1)
        self.fc_v = torch.nn.Linear(1024, 1)

        self.min_score = 1.0
        self.max_score = 7.0

    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # mean pooling
        attn = batch['attention_mask']
        sentence_representation = (output.last_hidden_state * attn.unsqueeze(-1)).sum(-2) / attn.sum(dim=-1).unsqueeze(-1)

        unbounded_mean = self.fc_m(sentence_representation)
        scaled_mean = self.min_score + (self.max_score - self.min_score) * torch.sigmoid(unbounded_mean)

        var = F.softplus(self.fc_v(sentence_representation)) # variance must be positive

        return scaled_mean.squeeze(), var.squeeze()
    
class LitPariedTextModel(L.LightningModule):
    def __init__(
        self,
        plm_names: list[str],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        log_dir: str,
        save_uc_metrics: bool,
        error_decay_factor: float,
        loss_weights: list[float] 
    ):
        super().__init__()
        self.save_hyperparameters()

        if len(plm_names) == 2:
            self.model = BiEncoderModel(plm_names=plm_names)
        elif len(plm_names) == 1:
            self.model = CrossEncoderModel(plm_name=plm_names[0])

        self.lr = lr
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.log_dir = log_dir
        self.save_uc_metrics = save_uc_metrics

        self.error_decay_factor = error_decay_factor
        self.loss_weights = loss_weights
        
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.0, 0.98),
            eps=1e-6,
            weight_decay=0.1
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
    
    def _compute_penalty_loss(self, mean, var, labels) -> torch.Tensor:
        errors = (mean - labels) ** 2

        # exponential decay weighting
        # When error is small, weight ~ exp(-alpha*small) ~ 1; when error is large, weight decays
        weight = torch.exp(-self.error_decay_factor * errors)

        var = var * weight

        penalty_loss = torch.linalg.norm(var, ord=2) / var.numel()

        return self.loss_weights[0] * penalty_loss
    
    def compute_and_log_loss(self, mean, var, labels, mode: str):
        nll_loss = F.gaussian_nll_loss(mean, labels.squeeze(), var)

        penalty_loss = self._compute_penalty_loss(mean=mean, var=var, labels=labels)
        
        total_loss = nll_loss + penalty_loss

        self.log(f"{mode}_nll_loss", nll_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{mode}_penalty_loss", penalty_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{mode}_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss
    
    def training_step(self, batch, batch_idx):
        mean, var = self(batch)
        loss = self.compute_and_log_loss(mean, var, batch["labels"], mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        mean, var = self(batch)
        _ = self.compute_and_log_loss(mean, var, batch["labels"], mode="val")

        self.validation_outputs.append({
            "mean": mean,
            "var": var,
            "labels": batch["labels"]
        })
    
    def on_validation_epoch_end(self):
        all_means = torch.cat([out["mean"] for out in self.validation_outputs])
        all_labels = torch.cat([out["labels"] for out in self.validation_outputs])

        all_means = all_means.to(torch.float64).cpu()
        all_labels = all_labels.to(torch.float64).cpu()

        self.log(
            "val_pcc",
            pearson_corrcoef(all_means, all_labels),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )

        self.log(
            "val_ccc",
            concordance_corrcoef(all_means, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "val_rmse",
            mean_squared_error(all_means, all_labels, squared=False),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )

        self.validation_outputs.clear()

    def _save_npy(self, output: list[dict], save_str: str = "unc"):
        # convert tensors to numpy arrays
        output_dict = {}
        for out in output:
            for key in out:
                array = out[key].cpu().numpy()
                if key in output_dict:
                    output_dict[key] = np.concatenate((output_dict[key], array), axis=0)
                else:
                    output_dict[key] = array
        
        np.save(f"{self.log_dir}/output_{save_str}.npy", output_dict)
        log_info(logger, f"Saved output to {self.log_dir}/output_{save_str}.npy")

    def test_step(self, batch, batch_idx):
        mean, var = self(batch)
        
        outputs = {
            "mean": mean,
            "var": var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"]
        
        if "noise" in batch:
            outputs["noise"] = batch["noise"]

        self.test_outputs.append(outputs)

    def on_test_epoch_end(self):
        all_means = torch.cat([out["mean"] for out in self.test_outputs])
        all_means = all_means.to(torch.float64).cpu()

        if "labels" in self.test_outputs[0]:
            all_labels = torch.cat([out["labels"] for out in self.test_outputs])
            all_labels = all_labels.to(torch.float64).cpu()

            self.log(
                "test_pcc",
                pearson_corrcoef(all_means, all_labels),
                logger=False,
                prog_bar=True,
                sync_dist=True
            )

            self.log(
                "test_ccc",
                concordance_corrcoef(all_means, all_labels),
                logger=False,
                prog_bar=True,
                sync_dist=True
            )

            self.log(
                "test_rmse",
                mean_squared_error(all_means, all_labels, squared=False),
                logger=False,
                prog_bar=True,
                sync_dist=True
            )

        if self.save_uc_metrics:
            self._save_npy(self.test_outputs)

        self.test_outputs.clear()

class PairedTextModelTrainer(object):
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
        do_tune: bool = False,
        do_test: bool = False,
        plm_names: list[str] = ["roberta-base"],
        error_decay_factor: float = 0.5,
        loss_weights: list[float] = [0]
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
        self.do_tune = do_tune
        self.do_test = do_test
        self.error_decay_factor = error_decay_factor
        self.loss_weights = loss_weights

        self.enable_early_stopping = True
        self.early_stopping_start_epoch = 5
        self.enable_checkpointing = True
        self.plm_names =plm_names
        self.save_uc_metrics = False
        self.warmup_ratio = 0.06

        self.dm = PairedTextDataModule(
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

    def _seed_wise_train_validate(
            self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None
        ) -> tuple[str, dict]:
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
        if os.path.exists(curr_log_dir) and not self.do_tune:
            log_info(logger, f"Seed-level logging directory already exists: {curr_log_dir}. So, validating on the saved ckpt...")
            ckpt_list = glob.glob(os.path.join(curr_log_dir, "**/*.ckpt"), recursive=True)
            assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
            best_model_ckpt = ckpt_list[0]
        else:
            log_info(logger, f"Training from scratch")  
            with trainer.init_module():
                # model created here directly goes to GPU
                model = LitPariedTextModel(
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
            model = LitPariedTextModel.load_from_checkpoint(best_model_ckpt)

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
            model = LitPariedTextModel.load_from_checkpoint(model_path, save_uc_metrics=self.save_uc_metrics)

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
            
            if self.do_test:
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

    def tune_train_test(self, n_trials: int, parent_log_dir: str, seeds: int = 0) -> None:
        if self.do_tune:
            optuna_log_dir = os.path.join(parent_log_dir, "optuna_logs")
            os.makedirs(optuna_log_dir, exist_ok=True)
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

            study.optimize(objective_params, n_trials=n_trials, show_progress_bar=False)

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
        
        self.train_test(seeds=seeds, parent_log_dir=parent_log_dir)