import os
import logging
from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as F
# import torch.distributions as dist
import lightning as L
from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.utilities import CombinedLoader
# import warnings

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation

from paired_texts_modelling import PairedTextModelController, CrossEncoderProbModel, LitPairedTextModel
from utils import log_info, beta_nll_loss

logger = logging.getLogger(__name__)

class LitTwoModels(LitPairedTextModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.approach == "cross-prob":
            # we already have self.model from the parent class.
            self.model_2 = CrossEncoderProbModel(plm_name=self.hparams.plm_names[1])
        else:
            raise ValueError(f"Invalid or not-implemented approach: {self.approach}")
        
        # self.hparams.error_decay_factor = None # No penalty is added here yet
        # self.penalty_type = None
        # self.train_means = []
        # self.train_vars = []

    def _compute_consistency_loss(self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor) -> Tensor:
        # dist_1 = dist.Normal(mean_1, torch.sqrt(var_1))
        # dist_2 = dist.Normal(mean_2, torch.sqrt(var_2))
        # # kl_div is not symmetric, so taking the mean of both ways
        # consistency = 0.5 * (
        #     dist.kl_divergence(dist_1, dist_2).mean() +
        #     dist.kl_divergence(dist_2, dist_1).mean()
        # )
        
        # 2-Wasserstein distance
        consistency = ((mean_1 - mean_2) ** 2 + (torch.sqrt(var_1) - torch.sqrt(var_2)) ** 2).mean()

        return consistency

    def _compute_loss(
            self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor, 
            sentence_rep_1: Tensor, sentence_rep_2: Tensor,
            labels: Tensor, prefix: str,
            input_ids_1: Tensor, input_ids_2: Tensor,
            hidden_state_1: Tensor, hidden_state_2: Tensor
        ) -> tuple[Tensor, dict]:
        loss_dict = {}
        
        # l2_var = torch.linalg.norm(var_1 - var_2, ord=2)

        # consistency = self.lambda_1 * F.mse_loss(var_1, var_2)

        # mean = 0.5 * (mean_1 + mean_2)
        # var = 0.5 * (var_1 + var_2)
        # # nll = F.gaussian_nll_loss(mean, labels.squeeze(), var)
        # nll = beta_nll_loss(mean, var, labels)

        # nll_1 = F.gaussian_nll_loss(mean_1, labels.squeeze(), var_1)
        # nll_2 = F.gaussian_nll_loss(mean_2, labels.squeeze(), var_2)
        # nll = 0.5 * (nll_1 + nll_2)

        avg_var = 0.5 * (var_1 + var_2)
        
        if self.lambdas[0] != 0:
            nll_1 = beta_nll_loss(mean_1, avg_var, labels)
            nll_2 = beta_nll_loss(mean_2, avg_var, labels)
            nll = self.lambdas[0] * 0.5 * (nll_1 + nll_2)
            loss_dict[f"{prefix}_nll"] = nll.item()
        else: 
            nll = 0

        if self.lambdas[1] != 0:
            penalty_1 = self._compute_penalty_loss(mean_1, avg_var, labels)
            penalty_2 = self._compute_penalty_loss(mean_2, avg_var, labels)
            penalty = self.lambdas[1] * 0.5 * (penalty_1 + penalty_2)
            loss_dict[f"{prefix}_penalty"] = penalty.item()
        else:
            penalty = 0

        if self.lambdas[2] != 0:
            consistency = self.lambdas[2] * self._compute_consistency_loss(mean_1, mean_2, var_1, var_2)
            loss_dict[f"{prefix}_consistency"] = consistency.item()
        else:
            consistency = 0
        
        if self.lambdas[3] != 0:
            repr_consistency = self.lambdas[3] * F.mse_loss(sentence_rep_1, sentence_rep_2)
            loss_dict[f"{prefix}_repr_consistency"] = repr_consistency.item()
        else:
            repr_consistency = 0

        if self.lambdas[4] != 0:
            alignment_1 = self._compute_alignment_betn_texts(
                input_ids=input_ids_1,
                hidden_state=hidden_state_1,
                labels=labels
            )
            alignment_2 = self._compute_alignment_betn_texts(
                input_ids=input_ids_2,
                hidden_state=hidden_state_2,
                labels=labels
            )
            alignment = self.lambdas[4] * 0.5 * (alignment_1 + alignment_2)
            loss_dict[f"{prefix}_loss_alignment"] = alignment.item()
        else:
            alignment = 0
        
        loss = nll + penalty + consistency + repr_consistency + alignment
        loss_dict[f"{prefix}_loss"] = loss.item()

        return loss, loss_dict

    # def _compute_loss_unlbl(
    #         self, mean_1: Tensor, mean_2: Tensor, 
    #         var_1: Tensor, var_2: Tensor,
    #         pseudo_lbl: Tensor, pseudo_var: Tensor,
    #         sentence_rep_1: Tensor, sentence_rep_2: Tensor,
    #         prefix: str
    #     ) -> tuple[Tensor, dict]:
    #     # cons_12 = F.kl_div(F.log_softmax(mean_1, dim=-1), F.softmax(mean_2, dim=-1), reduction="batchmean")
    #     # cons_21 = F.kl_div(F.log_softmax(mean_2, dim=-1), F.softmax(mean_1, dim=-1), reduction="batchmean")

    #     # l2_var = torch.linalg.norm(var_1 - var_2, ord=2)

    #     # mean = 0.5 * (mean_1 + mean_2)
    #     # var = 0.5 * (var_1 + var_2)
    #     # loss = F.kl_div(F.log_softmax(mean_1, dim=1), F.softmax(mean_2, dim=1), reduction="batchmean")
    #     # loss = F.gaussian_nll_loss(mean_1, mean, var) + F.gaussian_nll_loss(mean_2, mean, var)
    #     # consistency = self.lambda_2 * (F.mse_loss(var_1, var) + F.mse_loss(var_2, var))

    #     # nll_1 = torch.mean(torch.exp(-var_2) * F.gaussian_nll_loss(mean_1, mean_2, var_1, reduction="none"))
    #     # nll_2 = torch.mean(torch.exp(-var_1) * F.gaussian_nll_loss(mean_2, mean_1, var_2, reduction="none"))
    #     # loss = 0.5 * (nll_1 + nll_2)

    #     nll_1 = beta_nll_loss(mean_1, pseudo_var, pseudo_lbl)
    #     nll_2 = beta_nll_loss(mean_2, pseudo_var, pseudo_lbl)
    #     nll = 0.5 * (nll_1 + nll_2)

    #     var_mse_0 = F.mse_loss(var_1, pseudo_var)
    #     var_mse_1 = F.mse_loss(var_2, pseudo_var)
    #     var_mse = 0.5 * (var_mse_0 + var_mse_1)

    #     pseudo_supervision = self.lambda_2 * (nll + var_mse)

    #     # loss = var_consistency + nll

    #     # consistency = self.lambda_2 * self._compute_consistency_loss(mean_1, mean_2, var_1, var_2)
        
    #     repr_consistency = self.lambda_3 * F.mse_loss(sentence_rep_1, sentence_rep_2)

    #     # lbl_dist = dist.Normal(self.global_mean, torch.sqrt(self.global_var))
    #     # mean = 0.5 * (mean_1 + mean_2)
    #     # var = 0.5 * (var_1 + var_2)
    #     # unlbl_dist = dist.Normal(mean, torch.sqrt(var))
    #     # domain_gap = self.lambda_3 * dist.kl_divergence(unlbl_dist, lbl_dist).mean()

    #     # loss = consistency + domain_gap

    #     # loss = consistency + repr_consistency
    #     loss = pseudo_supervision + repr_consistency

    #     loss_dict = {
    #         # f"{prefix}_consistency": consistency,
    #         f"{prefix}_pseudo_supervision": pseudo_supervision.item(),
    #         # f"{prefix}_domain_gap": domain_gap,
    #         f"{prefix}_repr_consistency": repr_consistency.item(),
    #         f"{prefix}_loss": loss.item()
    #     }

    #     return loss, loss_dict

    def training_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s, hidden_1s, hidden_2s, sentence_rep_1s, sentence_rep_2s = [], [], [], [], [], [], [], []
        
        for _ in range(self.num_passes):
            mean_1, var_1, sentence_rep_1, hidden_state_1 = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, sentence_rep_2, hidden_state_2 = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            var_1s.append(var_1)
            mean_2s.append(mean_2)
            var_2s.append(var_2)
            sentence_rep_1s.append(sentence_rep_1)
            sentence_rep_2s.append(sentence_rep_2)
            hidden_1s.append(hidden_state_1)
            hidden_2s.append(hidden_state_2)
            
        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (num_passes, bsz) -> (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)
        sentence_rep_1 = torch.stack(sentence_rep_1s, dim=0).mean(dim=0)
        sentence_rep_2 = torch.stack(sentence_rep_2s, dim=0).mean(dim=0)
        hidden_state_1 = torch.stack(hidden_1s, dim=0).mean(dim=0)
        hidden_state_2 = torch.stack(hidden_2s, dim=0).mean(dim=0)

        total_loss, loss_dict = self._compute_loss(
            mean_1=mean_1, mean_2=mean_2, 
            var_1=var_1, var_2=var_2, 
            sentence_rep_1=sentence_rep_1, sentence_rep_2=sentence_rep_2,
            labels=batch["labels"], prefix="train_lbl",
            input_ids_1=batch["input_ids_1"],
            input_ids_2=batch["input_ids_2"],
            hidden_state_1=hidden_state_1,
            hidden_state_2=hidden_state_2
        )

        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
            batch_size=batch["labels"].shape[0] # assuming the batch size is same for lbl and unlbl
        )

        # mean = 0.5 * (mean_1_lbl + mean_2_lbl)
        # var = 0.5 * (var_1_lbl + var_2_lbl)
        # self.train_means.append(mean.detach().unsqueeze(0) if mean.dim() == 0 else mean.detach())
        # self.train_vars.append(var.detach().unsqueeze(0) if var.dim() == 0 else var.detach())
        
        return total_loss

    # def training_step_single_pass(self, batch: dict, batch_idx):

    #     batch_lbl = batch

    #     mean_1_lbl, var_1_lbl, _, hidden_state_1 = self.model(
    #         input_ids=batch_lbl["input_ids_1"],
    #         attention_mask=batch_lbl["attention_mask_1"]
    #     )
    #     mean_2_lbl, var_2_lbl, _, hidden_state_2 = self.model_2(
    #         input_ids=batch_lbl["input_ids_2"],
    #         attention_mask=batch_lbl["attention_mask_2"]
    #     )

    #     total_loss, loss_dict = self._compute_loss_lbl(
    #         mean_1=mean_1_lbl, mean_2=mean_2_lbl, 
    #         var_1=var_1_lbl, var_2=var_2_lbl, 
    #         # sentence_rep_1=sentence_rep_1_lbl, sentence_rep_2=sentence_rep_2_lbl,
    #         labels=batch_lbl["labels"], prefix="train_lbl",
    #         input_ids_1=batch_lbl["input_ids_1"],
    #         input_ids_2=batch_lbl["input_ids_2"],
    #         hidden_state_1=hidden_state_1,
    #         hidden_state_2=hidden_state_2
    #     )

    #     self.log_dict(
    #         loss_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
    #         batch_size=batch_lbl["labels"].shape[0] # assuming the batch size is same for lbl and unlbl
    #     )

    #     mean = 0.5 * (mean_1_lbl + mean_2_lbl)
    #     var = 0.5 * (var_1_lbl + var_2_lbl)
    #     self.train_means.append(mean.detach().unsqueeze(0) if mean.dim() == 0 else mean.detach())
    #     self.train_vars.append(var.detach().unsqueeze(0) if var.dim() == 0 else var.detach())
        
    #     return total_loss

    # def training_step_ssl(self, batch: dict, batch_idx):

    #     batch_lbl = batch["lbl"]

    #     mean_1_lbl, var_1_lbl, _, hidden_state_1 = self.model_1(
    #         input_ids=batch_lbl["input_ids_1"],
    #         attention_mask=batch_lbl["attention_mask_1"]
    #     )
    #     mean_2_lbl, var_2_lbl, _, hidden_state_2 = self.model_2(
    #         input_ids=batch_lbl["input_ids_2"],
    #         attention_mask=batch_lbl["attention_mask_2"]
    #     )

    #     total_loss, loss_dict = self._compute_loss_lbl(
    #         mean_1=mean_1_lbl, mean_2=mean_2_lbl, 
    #         var_1=var_1_lbl, var_2=var_2_lbl, 
    #         # sentence_rep_1=sentence_rep_1_lbl, sentence_rep_2=sentence_rep_2_lbl,
    #         labels=batch_lbl["labels"], prefix="train_lbl",
    #         input_ids_1=batch_lbl["input_ids_1"],
    #         input_ids_2=batch_lbl["input_ids_2"],
    #         hidden_state_1=hidden_state_1,
    #         hidden_state_2=hidden_state_2
    #     )

    #     # global_mean is set in on_train_epoch_end, so it means "first epoch"
    #     # plus, if all unlabelled loss weights are 0, then it won't be used
    #     # if hasattr(self, "global_mean") and not (self.lambda_2 == 0 and self.lambda_3 == 0):
    #     if not (self.lambda_2 == 0 and self.lambda_3 == 0):
    #         batch_unlbl = batch["unlbl"]

    #         mean_1_unlbl, var_1_unlbl, sentence_rep_1_unlbl, _ = self.model_1(
    #             input_ids=batch_unlbl["input_ids_1"],
    #             attention_mask=batch_unlbl["attention_mask_1"]
    #         )
    #         mean_2_unlbl, var_2_unlbl, sentence_rep_2_unlbl, _ = self.model_2(
    #             input_ids=batch_unlbl["input_ids_2"],
    #             attention_mask=batch_unlbl["attention_mask_2"]
    #         )

    #         mean_1s = []
    #         mean_2s = []
    #         var_1s = []
    #         var_2s = []

    #         for _ in range(self.num_passes):
    #             mean_1_, var_1_, _, _ = self.model_1(
    #                 input_ids=batch_unlbl["input_ids_1"],
    #                 attention_mask=batch_unlbl["attention_mask_1"]
    #             )
    #             mean_2_, var_2_, _, _ = self.model_2(
    #                 input_ids=batch_unlbl["input_ids_2"],
    #                 attention_mask=batch_unlbl["attention_mask_2"]
    #             )

    #             mean_1s.append(mean_1_)
    #             var_1s.append(var_1_)
    #             mean_2s.append(mean_2_)
    #             var_2s.append(var_2_)
            
    #         mean_1_ensemble = torch.stack(mean_1s, dim=0).mean(dim=0) # (num_passes, bsz) -> (bsz)
    #         mean_2_ensemble = torch.stack(mean_2s, dim=0).mean(dim=0)
    #         var_1_ensemble = torch.stack(var_1s, dim=0).mean(dim=0)
    #         var_2_ensemble = torch.stack(var_2s, dim=0).mean(dim=0)

    #         pseudo_lbl = 0.5 * (mean_1_ensemble + mean_2_ensemble)
    #         pseudo_var = 0.5 * (var_1_ensemble + var_2_ensemble)

    #         loss_unlbl, loss_dict_unlbl = self._compute_loss_unlbl(
    #             mean_1=mean_1_unlbl, mean_2=mean_2_unlbl, 
    #             var_1=var_1_unlbl, var_2=var_2_unlbl,
    #             pseudo_lbl=pseudo_lbl, pseudo_var=pseudo_var, 
    #             sentence_rep_1=sentence_rep_1_unlbl, 
    #             sentence_rep_2=sentence_rep_2_unlbl,
    #             prefix="train_unlbl"
    #         )

    #         total_loss += loss_unlbl

    #         loss_dict.update(loss_dict_unlbl)

    #         loss_dict["train_total_loss"] = total_loss.item()

    #     self.log_dict(
    #         loss_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
    #         batch_size=batch_lbl["labels"].shape[0] # assuming the batch size is same for lbl and unlbl
    #     )

    #     mean = 0.5 * (mean_1_lbl + mean_2_lbl)
    #     var = 0.5 * (var_1_lbl + var_2_lbl)
    #     self.train_means.append(mean.detach().unsqueeze(0) if mean.dim() == 0 else mean.detach())
    #     self.train_vars.append(var.detach().unsqueeze(0) if var.dim() == 0 else var.detach())
        
    #     return total_loss
    #     
    # def on_train_epoch_end(self):
    #     all_means = torch.cat(self.train_means, dim=0)
    #     all_vars = torch.cat(self.train_vars, dim=0)

    #     # self.global_mean = all_means.mean()

    #     # epistemic uncertainty + aleatoric uncertainty
    #     self.global_var = all_means.var(unbiased=False) + all_vars.mean()

    #     self.train_means.clear()
    #     self.train_vars.clear()

    def _enable_dropout_at_inference(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        for m in self.model_2.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    
    def validation_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s, hidden_1s, hidden_2s, sentence_rep_1s, sentence_rep_2s = [], [], [], [], [], [], [], []

        if self.num_passes > 1:
            self._enable_dropout_at_inference()
        
        for _ in range(self.num_passes):
            
            mean_1, var_1, sentence_rep_1, hidden_state_1 = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, sentence_rep_2, hidden_state_2 = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            mean_2s.append(mean_2)
            var_1s.append(var_1)
            var_2s.append(var_2)
            hidden_1s.append(hidden_state_1)
            hidden_2s.append(hidden_state_2)
            sentence_rep_1s.append(sentence_rep_1)
            sentence_rep_2s.append(sentence_rep_2)

        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)
        sentence_rep_1 = torch.stack(sentence_rep_1s, dim=0).mean(dim=0)
        sentence_rep_2 = torch.stack(sentence_rep_2s, dim=0).mean(dim=0)
        hidden_state_1 = torch.stack(hidden_1s, dim=0).mean(dim=0)
        hidden_state_2 = torch.stack(hidden_2s, dim=0).mean(dim=0)

        _, loss_dict = self._compute_loss(
            mean_1=mean_1, mean_2=mean_2,
            var_1=var_1, var_2=var_2,
            sentence_rep_1=sentence_rep_1, sentence_rep_2=sentence_rep_2,
            labels=batch["labels"], prefix="val",
            input_ids_1=batch["input_ids_1"],
            input_ids_2=batch["input_ids_2"],
            hidden_state_1=hidden_state_1,
            hidden_state_2=hidden_state_2
        )
        
        self.log_dict(
            loss_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
            batch_size=batch["labels"].shape[0]
        )

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        self.validation_outputs.append({
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var,
            "labels": batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]
        })

    def test_step(self, batch, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []

        if self.num_passes > 1:
            self._enable_dropout_at_inference()

        for _ in range(self.num_passes):
            
            mean_1, var_1, _, _ = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, _, _ = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            mean_2s.append(mean_2)
            var_1s.append(var_1)
            var_2s.append(var_2)

        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        outputs = {
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]

        self.test_outputs.append(outputs)

class LitUCVME(LitTwoModels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _compute_loss(
            self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor,
            labels: Tensor, prefix: str
        ) -> tuple[Tensor, dict]:
        loss_dict = {}

        avg_var = 0.5 * (var_1 + var_2)
        
        nll_1 = F.gaussian_nll_loss(mean_1, labels.squeeze(), avg_var)
        nll_2 = F.gaussian_nll_loss(mean_2, labels.squeeze(), avg_var)
        nll = nll_1 + nll_2
        loss_dict[f"{prefix}_nll"] = nll.item()

        consistency_1 = F.mse_loss(var_1, avg_var)
        consistency_2 = F.mse_loss(var_2, avg_var)
        consistency = consistency_1 + consistency_2
        loss_dict[f"{prefix}_consistency"] = consistency.item()
        
        loss = nll + consistency
        loss_dict[f"{prefix}_loss"] = loss.item()

        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []
        with torch.no_grad():
            for _ in range(self.num_passes):
                mean_1, var_1, _, _ = self.model(
                    input_ids=batch["input_ids_1"],
                    attention_mask=batch["attention_mask_1"]
                )
                mean_2, var_2, _, _ = self.model_2(
                    input_ids=batch["input_ids_2"],
                    attention_mask=batch["attention_mask_2"]
                )

                mean_1s.append(mean_1)
                var_1s.append(var_1)
                mean_2s.append(mean_2)
                var_2s.append(var_2)
            
        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (num_passes, bsz) -> (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        loss_reg_cps, _ = self._compute_loss(
            mean_1=mean_1, mean_2=mean_2, 
            var_1=var_1, var_2=var_2, 
            labels=batch["labels"], prefix="train_lbl"
        )

        outcome = batch["labels"]

        mean_1, var_1, _, _ = self.model(
            input_ids=batch["input_ids_1"],
            attention_mask=batch["attention_mask_1"]
        )
        mean_2, var_2, _, _ = self.model_2(
            input_ids=batch["input_ids_2"],
            attention_mask=batch["attention_mask_2"]
        )

        # we are not scaling-rescaling the labels, so no y_mean and y_std
        loss_mse_1 = F.mse_loss(mean_1, outcome)
        loss1 = torch.mul(torch.exp(-(var_1 + var_2) / 2), loss_mse_1)
        loss2 = (var_1 + var_2) / 2
        loss_1 = .5 * (loss1 + loss2)

        loss_reg_1 = loss_1.mean()

        loss_mse_2 = F.mse_loss(mean_2, outcome)
        loss1 = torch.mul(torch.exp(-(var_1 + var_2) / 2), loss_mse_2)
        loss2 = (var_1 + var_2) / 2
        loss_2 = .5 * (loss1 + loss2)

        loss_reg_2 = loss_2.mean()

        loss_reg = (loss_reg_1 + loss_reg_2)
        loss = loss_reg + self.lambdas[0] * loss_reg_cps +  ((var_2 - var_1) ** 2).mean()
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []

        if self.num_passes > 1:
            self._enable_dropout_at_inference()
        
        for _ in range(self.num_passes):
            
            mean_1, var_1, _, _ = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, _, _ = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            mean_2s.append(mean_2)
            var_1s.append(var_1)
            var_2s.append(var_2)

        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        self.validation_outputs.append({
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var,
            "labels": batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]
        })

    
class TwoModelsController(PairedTextModelController):
    def __init__(
            self,
            # unlbl_data_files: list[str],
            is_ucvme: bool,
            *args, **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        # self.unlbl_data_files = unlbl_data_files
        self.is_ucvme = is_ucvme
        if self.is_ucvme:
            self.model_class = LitUCVME
        else:
            self.model_class = LitTwoModels

    def _seed_wise_train_validate(self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None) -> tuple[str, dict]:
        L.seed_everything(seed)

        train_dl = self.dm.get_train_dl(
            data_path_list=self.train_file,
            batch_size=self.train_bsz,
            sanitise_newsemp_labels=False,
            add_noise=False,
            seed=seed,
            is_newsemp=self.is_newsemp_main,
            lbl_split=self.lbl_split,
            do_augment=self.do_augment
        )
        
        # In-domain SSL
        # train_dl_lbl, train_dl_unlbl = self.dm.get_ssl_dls(
        #     data_paths=self.train_file,
        #     sanitise_newsemp_labels=False,
        #     add_noise=False,
        #     is_newsemp=self.is_newsemp_main,
        #     lbl_split=self.lbl_split,
        #     seed=seed,
        #     batch_size=self.train_bsz
        # )
        
        # Out-of-domain SSL
        # train_dl_lbl = self.dm.get_train_dl(
        #     data_path_list=self.train_file,
        #     batch_size=self.train_bsz,
        #     sanitise_newsemp_labels=True,
        #     add_noise=False,
        #     seed=seed,
        #     is_newsemp=self.is_newsemp_main
        # )
        # train_dl_unlbl = self.dm.get_train_dl(
        #     data_path_list=self.unlbl_data_files,
        #     batch_size=self.train_bsz,
        #     sanitise_newsemp_labels=False,
        #     add_noise=False,
        #     seed=seed,
        #     is_newsemp=not self.is_newsemp_main # opposite of the main data
        # )

        # train_dl = CombinedLoader({"lbl": train_dl_lbl, "unlbl": train_dl_unlbl}, mode="max_size_cycle")

        trainer = self._prepare_trainer(curr_log_dir=curr_log_dir, extra_callbacks=extra_callbacks)
        
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        if os.path.exists(curr_log_dir) and not self.do_tune:
            log_info(logger, f"Seed-level logging directory already exists: {curr_log_dir}. So, validating on the saved ckpt...")
            ckpt_list = list(Path(curr_log_dir).rglob("*.ckpt"))
            assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
            best_model_ckpt = ckpt_list[0]
        else:
            log_info(logger, f"Training ...")  
            with trainer.init_module():
                # model created here directly goes to GPU
                model = self.model_class(
                    num_passes=self.num_passes,
                    plm_names=self.plm_names,
                    lr=self.lr,
                    log_dir=curr_log_dir,
                    save_uc_metrics=self.save_uc_metrics,
                    error_decay_factor=self.error_decay_factor,
                    lambdas=self.lambdas,
                    approach=self.approach,
                    sep_token_id=self.dm.tokeniser.sep_token_id
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
            model = self.model_class.load_from_checkpoint(best_model_ckpt)

        trainer.validate(model=model, dataloaders=self.val_dl)

        metrics = {key: value.item() for key, value in trainer.callback_metrics.items()}

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

        # experimental - stop the trainer
        # if trainer.is_global_zero:
        del trainer
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return best_model_ckpt, metrics

    def evaluate(self, model_path: str):
        tester = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        with tester.init_module(empty_init=True):
            model = self.model_class.load_from_checkpoint(model_path)

        tester.test(model=model, dataloaders=self.test_dl, verbose=True)

        metrics = {key: value.item() for key, value in tester.callback_metrics.items()}
        
        return metrics

    def optuna_objective(self, trial: optuna.trial.Trial, optuna_seed: int, optuna_log_dir: str) -> float:
        # self.lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 4e-5])
        # self.train_bsz = trial.suggest_int("train_bsz", 8, 32, step=8)

        # self.num_passes = trial.suggest_int("num_passes", 1, 4)

        if self.is_ucvme:
            # only one hp
            self.lambdas = [trial.suggest_float("lambda_0", 0.0, 50.0)]
        else:            
            lambda_1 = trial.suggest_float("lambda_1", 0.0, 50.0)
            lambda_2 = trial.suggest_float("lambda_2", 0.0, 50.0)
            lambda_3 = trial.suggest_float("lambda_3", 0.0, 50.0)
            lambda_4 = trial.suggest_float("lambda_4", 0.0, 50.0)
            self.lambdas = [1.0, lambda_1, lambda_2, lambda_3, lambda_4]

            # self.error_decay_factor = trial.suggest_float("error_decay_factor", 0.0, 3.0, step=0.5)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        _, metrics = self._seed_wise_train_validate(
            seed=optuna_seed,
            curr_log_dir=optuna_log_dir,
            extra_callbacks=[pruning_callback]
        )

        return metrics["val_ccc"]
    