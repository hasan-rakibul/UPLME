from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import torch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error
import os
import logging
import pandas as pd
import zipfile
from omegaconf import OmegaConf
from utils import log_info

logger = logging.getLogger(__name__)
    
class LightningPLM(L.LightningModule):
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
 
def init_model(config: OmegaConf, lr: float) -> L.LightningModule:
    return LightningPLM(config, lr=lr)
    
def load_model_from_ckpt(config: OmegaConf, ckpt: str) -> L.LightningModule:
    return LightningPLM.load_from_checkpoint(ckpt, config=config)
 