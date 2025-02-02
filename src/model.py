import torch
import torch.nn.functional as F
import lightning as L
from transformers import (
    AutoModel, AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
)
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import zipfile
from omegaconf import OmegaConf
from utils import log_info

logger = logging.getLogger(__name__)

class ProbabilisticPLM(torch.nn.Module):
    def __init__(
        self,
        plm_name: str,
        lr: float
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        self.fc_m = torch.nn.Linear(768, 1)
        self.fc_v = torch.nn.Linear(768, 1)

        self.lr = lr

        self.min_score = 1.0
        self.max_score = 7.0

    def forward(self, batch):
        output = self.plm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        x = output.last_hidden_state[:, 0, :]

        unbounded_mean = self.fc_m(torch.nn.functional.dropout(x, p=0.25))
        scaled_mean = self.min_score + (self.max_score - self.min_score) * torch.nn.functional.sigmoid(unbounded_mean)
        
        var = torch.nn.functional.softplus(self.fc_v(x)) # variance must be positive

        return scaled_mean.squeeze(), var.squeeze()

class LitProbabilisticPLMSingle(L.LightningModule):
    def __init__(
        self,
        plm_names: list[str],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        log_dir: str,
        save_uc_metrics: bool
    ):
        super().__init__()
        self.save_hyperparameters()

        if len(plm_names) == 1:
            self.model = ProbabilisticPLM(plm_name=plm_names[0], lr=lr)

        self.lr = lr
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.log_dir = log_dir
        self.save_uc_metrics = save_uc_metrics
        
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
    
    def compute_and_log_loss(self, mean, var, labels, mode):
        nll_loss = F.gaussian_nll_loss(mean, labels.squeeze(), var)

        self.log(f"{mode}_loss", nll_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return nll_loss
    
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

class LitProbabilisticPLMEnsemble(LitProbabilisticPLMSingle):
    def __init__(
        self,
        plm_names: list[str],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        loss_weights: list[float],
        log_dir: str,
        save_uc_metrics: bool
    ):
        super().__init__(
            plm_names=plm_names,
            lr=lr,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            log_dir=log_dir,
            save_uc_metrics=save_uc_metrics
        )

        self.save_hyperparameters()

        self.model = torch.nn.ModuleList([
            ProbabilisticPLM(plm_name=plm, lr=lr) for plm in plm_names
        ])
        
        self.loss_weights = loss_weights
        assert len(self.loss_weights) == 2, "Check if the number of loss weights is correct"

        self.model_weights = torch.nn.Parameter(
            torch.ones(len(plm_names)) / len(plm_names),
            requires_grad=True
        )

    def forward(self, batch):
        means = []
        variances = []
        for sub_model in self.model:
            mean, var = sub_model(batch)
            means.append(mean)
            variances.append(var)
        means_tensor = torch.stack(means) # (num_models, batch_size)
        variances_tensor = torch.stack(variances)

        # weighted ensemble
        weights = F.softmax(self.model_weights, dim=0) # (num_models,)

        ensemble_mean = torch.sum(means_tensor * weights.unsqueeze(-1), dim=0) # (batch_size,)
        ensemble_var = torch.sum(variances_tensor * (weights.unsqueeze(-1)**2), dim=0)

        return ensemble_mean, ensemble_var, variances_tensor
    
    def compute_and_log_loss(self, ensemble_mean, ensemble_var, labels, variances, mode):
        nll_loss = F.gaussian_nll_loss(ensemble_mean.squeeze(), labels.squeeze(), ensemble_var.squeeze())

        # simple consistency of ensemble_var
        # z = torch.log(ensemble_var)
        # consistency_loss = self.consistency_weight * torch.mean((z - torch.mean(z))**2)

        # consistency loss like UCVME but for any number of models
        consistency_losses = []
        variances = torch.clamp(variances, min=1e-8)
        for i in range(len(variances)):
            for j in range(i+1, len(variances)):
                consistency_loss = torch.mean((torch.log(variances[i]) - torch.log(variances[j]))**2)
                consistency_losses.append(consistency_loss)

        consistency_loss = self.loss_weights[0] * torch.mean(torch.stack(consistency_losses))

        # simple L2 penalty
        # variance_penalty_loss = self.penalty_weight * torch.norm(ensemble_var, p=2)

        # model-level variance penalty
        variance_penalty_losses = []
        for var in variances:
            variance_penalty_loss = torch.norm(var, p=2)
            variance_penalty_losses.append(variance_penalty_loss)

        variance_penalty_loss = self.loss_weights[1] * torch.mean(torch.stack(variance_penalty_losses))

        total_loss = nll_loss + consistency_loss + variance_penalty_loss

        self.log(f"{mode}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{mode}_nll_loss", nll_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{mode}_consistency_loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{mode}_penalty_loss", variance_penalty_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss
    
    def training_step(self, batch, batch_idx):
        ensemble_mean, ensemble_var, indiv_var = self(batch)
        loss = self.compute_and_log_loss(ensemble_mean, ensemble_var, batch["labels"], indiv_var, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        ensemble_mean, ensemble_var, indiv_var = self(batch)
        _ = self.compute_and_log_loss(ensemble_mean, ensemble_var, batch["labels"], indiv_var, mode="val")

        self.validation_outputs.append({
            "mean": ensemble_mean,
            "var": ensemble_var,
            "labels": batch["labels"]
        })

    def test_step(self, batch, batch_idx):
        ensemble_mean, ensemble_var, _ = self(batch)

        outputs = {
            "mean": ensemble_mean,
            "var": ensemble_var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"]
        
        if "noise" in batch:
            outputs["noise"] = batch["noise"]

        self.test_outputs.append(outputs)
        

class LitBasicPLM(L.LightningModule):
    def __init__(self, config: OmegaConf, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.lr = lr

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.config.plm,
            num_labels=1,
            # ignore_mismatched_sizes=True # SieBERT has 2 output labels so we ignore mismatched sizes
        )

        self.training_step_outputs = []
        self.training_step_labels = []
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        return output.logits.squeeze(-1) # remove the last dimension

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.config.adamw_beta1, self.config.adamw_beta2),
            eps=self.config.adamw_eps,
            weight_decay=self.config.adamw_weight_decay
        )

        if not self.config.lr_scheduler_type: # no lr scheduler
            return optimiser
        
        elif self.config.lr_scheduler_type == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode='min',
                patience=self.config.plateau_patience,
                factor=self.config.plateau_factor,
                threshold=self.config.plateau_threshold
            )
        elif self.config.lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimiser,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps
            )
        elif self.config.lr_scheduler_type == "polynomial":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimiser,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps,
                lr_end=1.0e-6
            )
        
        return {
            'optimizer': optimiser,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    @rank_zero_only
    def _calc_save_predictions(self, preds, mode='test'):
        preds_np = preds.numpy()
        
        pred_df = pd.DataFrame({'emp': preds_np, 'dis': preds_np}) # we're not predicting distress, just aligning with submission system
        pred_df.to_csv(
            os.path.join(self.config.logging_dir, f"{mode}-predictions_EMP.tsv"),
            sep='\t', index=None, header=None
        )
        log_info(logger, f'Saved predictions to {self.config.logging_dir}/{mode}-predictions_EMP.tsv')
        
        if "test_from_ckpts_parent_dir" in self.config:
            # need to reconfigure the logging dir for submission files
            zip_save_dir = os.path.join(self.config.logging_dir, f"seed_{self.config.seed}")
            os.makedirs(zip_save_dir, exist_ok=True)
        else:
            zip_save_dir = self.config.logging_dir

        if self.config.make_ready_for_submission:
            with zipfile.ZipFile(f"{zip_save_dir}/predictions.zip", "w") as zf:
                zf.write(f"{self.config.logging_dir}/{mode}-predictions_EMP.tsv", arcname="predictions_EMP.tsv")
                log_info(logger, f"Zipped predictions to {self.config.logging_dir}/predictions.zip")
            # remove the file as we have zipped it and the raw file doesn't matter anymore
            os.remove(f"{self.config.logging_dir}/{mode}-predictions_EMP.tsv")
        
    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = torch.nn.functional.mse_loss(outputs, batch['labels'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        self.training_step_outputs.append(outputs)
        self.training_step_labels.append(batch['labels'])

        return loss
    
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_step_outputs)
        all_labels = torch.cat(self.training_step_labels)

        # float32 (default) has poorer performance in pcc and ccc
        # metrics calculation in GPU widely varies from CPU
        # also, class version of metrics widely varies from functional version
        # functional version matched with scipy, numpy and WASSA organisers' evaluation
            
        all_preds = all_preds.to(torch.float64).cpu()
        all_labels = all_labels.to(torch.float64).cpu()

        self.log(
            'train_pcc', 
            pearson_corrcoef(all_preds, all_labels),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            'train_ccc',
            concordance_corrcoef(all_preds, all_labels),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            'train_rmse',
            mean_squared_error(all_preds, all_labels, squared=False),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )
        self.training_step_outputs.clear() # free up memory
        self.training_step_labels.clear()
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = torch.nn.functional.mse_loss(outputs, batch['labels'])
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

        self.validation_step_outputs.append(outputs)
        self.validation_step_labels.append(batch['labels'])
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_labels = torch.cat(self.validation_step_labels)

        all_preds = all_preds.to(torch.float64).cpu()
        all_labels = all_labels.to(torch.float64).cpu()

        self.log(
            'val_pcc',
            pearson_corrcoef(all_preds, all_labels),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            'val_ccc',
            concordance_corrcoef(all_preds, all_labels),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            'val_rmse',
            mean_squared_error(all_preds, all_labels, squared=False),
            logger=True,
            prog_bar=False,
            sync_dist=True
        )

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    # cannot be made to run on a single GPU
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        self.test_step_outputs.append(outputs)
        if 'labels' in batch:
            self.test_step_labels.append(batch['labels'])
    
    @rank_zero_only
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs)

        all_preds = all_preds.to(torch.float64).cpu()

        if len(self.test_step_labels) > 0:
            all_labels = torch.cat(self.test_step_labels)
            all_labels = all_labels.to(torch.float64).cpu()
        else:
            all_labels = None

        if all_labels is not None:
            self.log(
                'test_pcc',
                pearson_corrcoef(all_preds, all_labels),
                logger=False,
                prog_bar=False,
                sync_dist=True
            )
            self.log(
                'test_ccc',
                concordance_corrcoef(all_preds, all_labels),
                logger=False,
                prog_bar=False,
                sync_dist=True
            )
            self.log(
                'test_rmse',
                mean_squared_error(all_preds, all_labels, squared=False),
                logger=False,
                prog_bar=False,
                sync_dist=True
            )

        if self.config.save_predictions_to_disk:
            self._calc_save_predictions(all_preds, mode=f"{self.config.test_split}_{self.config.test_data[0]}")

        self.test_step_outputs.clear()
        self.test_step_labels.clear()
