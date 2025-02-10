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
from utils import log_info

logger = logging.getLogger(__name__)

class BiEncoder(torch.nn.Module):
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
        scaled_mean = self.min_score + (self.max_score - self.min_score) * F.sigmoid(unbounded_mean)
        
        var = F.softplus(self.fc_v(x)) # variance must be positive

        return scaled_mean.squeeze(), var.squeeze()
    
class LitBiEncoder(L.LightningModule):
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

        self.model = BiEncoder(plm_names=plm_names)

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
