import os
import logging
import glob
import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import CombinedLoader
import warnings

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation

from paired_texts_modelling import PairedTextModelController, CrossEncoderProbModel, LitPairedTextModel
from utils import log_info

logger = logging.getLogger(__name__)

class LitSSLModel(LitPairedTextModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.approach == "cross-prob":
            self.model_1 = self.model
            self.model_2 = CrossEncoderProbModel(plm_name=self.hparams.plm_names[1])
        else:
            raise ValueError(f"Invalid or not-implemented approach: {self.approach}")
        
        # self.hparams.error_decay_factor = None # No penalty is added here yet
        # self.penalty_type = None

    def _compute_and_log_loss_lbl(
            self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor, 
            labels: Tensor, prefix: str
        ) -> Tensor:
        # l2_var = torch.linalg.norm(var_1 - var_2, ord=2)

        # var_consistency = F.mse_loss(var_1, var_2)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)
        nll = F.gaussian_nll_loss(mean, labels.squeeze(), var)
        
        # nll_1 = F.gaussian_nll_loss(mean_1, labels.squeeze(), var_1)
        # nll_2 = F.gaussian_nll_loss(mean_2, labels.squeeze(), var_2)
        # nll = 0.5 * (nll_1 + nll_2)

        # var_consistency = self._compute_penalty_loss(mean, var, labels)

        # loss = var_consistency + nll
        loss = nll

        loss_dict = {
            # f"{prefix}_var_consistency": var_consistency,
            # f"{prefix}_nll": nll,
            f"{prefix}_loss": loss
        }
        self.log_dict(
            loss_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
            batch_size=labels.shape[0]
        )

        return loss

    def _compute_and_log_loss_unlbl(
            self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor, 
            prefix: str
        ) -> Tensor:
        # cons_12 = F.kl_div(F.log_softmax(mean_1, dim=-1), F.softmax(mean_2, dim=-1), reduction="batchmean")
        # cons_21 = F.kl_div(F.log_softmax(mean_2, dim=-1), F.softmax(mean_1, dim=-1), reduction="batchmean")

        # l2_var = torch.linalg.norm(var_1 - var_2, ord=2)

        # mean = 0.5 * (mean_1 + mean_2)
        # var = 0.5 * (var_1 + var_2)
        # loss = F.kl_div(F.log_softmax(mean_1, dim=1), F.softmax(mean_2, dim=1), reduction="batchmean")
        # loss = F.gaussian_nll_loss(mean_1, mean, var) + F.gaussian_nll_loss(mean_2, mean, var)
        # var_consistency = F.mse_loss(var_1, avg_var) + F.mse_loss(var_2, avg_var)

        nll_1 = torch.mean(torch.exp(-var_2) * F.gaussian_nll_loss(mean_1, mean_2, var_1, reduction="none"))
        nll_2 = torch.mean(torch.exp(-var_1) * F.gaussian_nll_loss(mean_2, mean_1, var_2, reduction="none"))
        loss = 0.5 * (nll_1 + nll_2)

        # if nll.detach().item() < 0:
        #     import pdb; pdb.set_trace()
        # loss = var_consistency + nll

        loss_dict = {
            # f"{prefix}_var_consistency": var_consistency,
            # f"{prefix}_nll": nll,
            f"{prefix}_loss": loss
        }
        self.log_dict(
            loss_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
            batch_size=mean_1.shape[0]
        )

        return loss

    def training_step(self, batch: dict, batch_idx):
        batch_lbl = batch["lbl"] 
        batch_unlbl = batch["unlbl"]

        mean_1_lbl, var_1_lbl = self.model_1(batch_lbl)
        mean_2_lbl, var_2_lbl = self.model_2(batch_lbl)
        mean_1_unlbl, var_1_unlbl = self.model_1(batch_unlbl)
        mean_2_unlbl, var_2_unlbl = self.model_2(batch_unlbl)

        loss_lbl = self._compute_and_log_loss_lbl(
            mean_1=mean_1_lbl, mean_2=mean_2_lbl, 
            var_1=var_1_lbl, var_2=var_2_lbl, 
            labels=batch_lbl["labels"], prefix="train_lbl"
        )
        
        loss_unlbl = self._compute_and_log_loss_unlbl(
            mean_1=mean_1_unlbl, mean_2=mean_2_unlbl, 
            var_1=var_1_unlbl, var_2=var_2_unlbl, 
            prefix="train_unlb"
        )

        loss = loss_lbl + self.loss_weight * loss_unlbl
        self.log(
            "train_total_loss", loss, 
            on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
            batch_size=batch_lbl["labels"].shape[0]
        )
        return loss
    
    def validation_step(self, batch: dict, batch_idx):
        mean_1, var_1 = self.model_1(batch)
        mean_2, var_2 = self.model_2(batch)

        _ = self._compute_and_log_loss_lbl(
            mean_1=mean_1, mean_2=mean_2, 
            var_1=var_1, var_2=var_2, 
            labels=batch["labels"], prefix="val"
        )

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)
        
        self.validation_outputs.append({
            "mean": mean,
            "var": var,
            "labels": batch["labels"]
        })

    def test_step(self, batch, batch_idx):
        mean_1, var_1 = self.model_1(batch)
        mean_2, var_2 = self.model_2(batch)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        outputs = {
            "mean": mean,
            "var": var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"]

        self.test_outputs.append(outputs)

class SSLModelController(PairedTextModelController):
    def __init__(
            self,
            unlbl_data_files: list[str],
            *args, **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.unlbl_data_files = unlbl_data_files

        if self.approach == "cross-prob":
            self.plm_names = ["roberta-base", "roberta-base"]
            if self.plm_names[0] != self.plm_names[1]:
                warnings.warn(f"Note that only {self.plm_names[0]} is used in tokenisation.")
    
    def _seed_wise_train_validate(self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None) -> tuple[str, dict]:
        L.seed_everything(seed)
        
        train_dl_lbl = self.dm.get_train_dl(
            data_path_list=self.train_file,
            batch_size=self.train_bsz,
            sanitise_newsemp_labels=True,
            add_noise=False,
            seed=seed,
            is_newsemp=self.is_newsemp_main
        )
        train_dl_unlbl = self.dm.get_train_dl(
            data_path_list=self.unlbl_data_files,
            batch_size=self.train_bsz,
            sanitise_newsemp_labels=False,
            add_noise=False,
            seed=seed,
            is_newsemp=not self.is_newsemp_main # opposite of the main data
        )

        train_dl = CombinedLoader({"lbl": train_dl_lbl, "unlbl": train_dl_unlbl}, mode="max_size_cycle")

        trainer = self._prepare_trainer(curr_log_dir=curr_log_dir, extra_callbacks=extra_callbacks)
        
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        if os.path.exists(curr_log_dir) and not self.do_tune:
            log_info(logger, f"Seed-level logging directory already exists: {curr_log_dir}. So, validating on the saved ckpt...")
            ckpt_list = glob.glob(os.path.join(curr_log_dir, "**/*.ckpt"), recursive=True)
            assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
            best_model_ckpt = ckpt_list[0]
        else:
            log_info(logger, f"Training ...")  
            with trainer.init_module():
                # model created here directly goes to GPU
                model = LitSSLModel(
                    plm_names=self.plm_names,
                    lr=self.lr,
                    log_dir=curr_log_dir,
                    save_uc_metrics=self.save_uc_metrics,
                    error_decay_factor=self.error_decay_factor,
                    loss_weight=self.loss_weight,
                    approach=self.approach
                )
            
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=self.val_dl
            )

            # getting the best model from the trainer
            best_model_ckpt = trainer.checkpoint_callback.best_model_path
        
        # final validation at the end of training
        log_info(logger, f"Loading the best model from {best_model_ckpt}")
        with trainer.init_module(empty_init=True):
            model = LitSSLModel.load_from_checkpoint(best_model_ckpt)

        trainer.validate(model=model, dataloaders=self.val_dl)

        metrics = {key: value.item() for key, value in trainer.callback_metrics.items()}

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

        return best_model_ckpt, metrics

    def evaluate(self, model_path: str):
        tester = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        with tester.init_module(empty_init=True):
            model = LitSSLModel.load_from_checkpoint(model_path)

        tester.test(model=model, dataloaders=self.test_dl, verbose=True)

        metrics = {key: value.item() for key, value in tester.callback_metrics.items()}
        
        return metrics

    def optuna_objective(self, trial: optuna.trial.Trial, optuna_seed: int, optuna_log_dir: str) -> float:
        self.lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 4e-5])
        self.train_bsz = trial.suggest_int("train_bsz", 8, 32, step=8)

        self.loss_weight = trial.suggest_float("loss_weight", 0.0, 100.0)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        _, metrics = self._seed_wise_train_validate(
            seed=optuna_seed,
            curr_log_dir=optuna_log_dir,
            extra_callbacks=[pruning_callback]
        )

        return metrics["val_ccc"]
    