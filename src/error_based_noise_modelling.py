import os
import numpy as np
import pandas as pd
import torch
import logging
import lightning as L

from utils import log_info, get_trainer, resolve_num_steps
from preprocess import DataModuleFromRaw
from model import init_model, load_model_from_ckpt

logger = logging.getLogger(__name__)

def _calculate_error(model, train_dl):
    # basic pytorch as Lightning predictions become distributed across gpus
    device = model.device
    model.eval()
    
    error_list = []
    sample_ids_list = []
    with torch.no_grad():
        for batch in train_dl:
            preds = model(batch.to(device))
            preds = preds.cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            error = (preds - labels) ** 2
            error_list.extend(error) # array can be appended to a list, it becomes a normal list
            sample_ids_list.extend(batch["sample_id"].cpu().tolist())

    return error_list, sample_ids_list

def _find_noisy_samples(config, sample_list: list, delta: float, lr: float):
    error_df = pd.DataFrame(index=sample_list)

    raw_logging_dir = config.logging_dir # so that we don't nest
    # for i, model in enumerate(models):
    for i in range(config.num_agents):
        config.seed = config.seeds[i]
        L.seed_everything(config.seed)

        # customise logging_dir per agent
        config.logging_dir = os.path.join(raw_logging_dir, f"agent_{i}")
        
        datamodule = DataModuleFromRaw(config, delta=delta, seed=config.seed)
        train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
        val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)
        
        model = init_model(config, lr=lr)

        trainer = get_trainer(config)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        
        model = load_model_from_ckpt(config=config, ckpt=trainer.checkpoint_callback.best_model_path)

        error, sample_id = _calculate_error(model, train_dl)
        error_df.loc[sample_id, f"error_{i}"] = error
        
    threshold = error_df.quantile(q=(1-config.noise_level)) 
    # 1-noise_level, as hc samples are below the threshold
    log_info(logger, f"Threshold:\n{threshold}")

    threshold_matrix = error_df < threshold # will be True if error < threshold for each sample

    # keep only the samples that are True for all columns;
    # False in any column will result in NaN, and thus the entire row is dropped
    hc_sample_ids = error_df[threshold_matrix].dropna().index.to_numpy()
    lc_sample_ids = error_df[~threshold_matrix].dropna().index.to_numpy()

    # if there are any samples that are not classified as HC or LC, they will be classified as MC
    mc_sample_ids = error_df.index.difference(hc_sample_ids).difference(lc_sample_ids).to_numpy()
    
    config.logging_dir = raw_logging_dir

    return hc_sample_ids, mc_sample_ids, lc_sample_ids

def noise_removal(config, delta: float, seed: int, lr: float):
    config.seed = config.seeds[0] # just need for train_dl, that will be updated
    datamodule = DataModuleFromRaw(config, delta=delta, seed=seed)
    train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)

    if config.lr_scheduler_type == "linear" or config.lr_scheduler_type == "polynomial":
        config.num_training_steps, config.num_warmup_steps = resolve_num_steps(config, train_dl)

    sample_list = train_dl.dataset["sample_id"].cpu().tolist()
    hc_sample_ids, mc_sample_ids, lc_sample_ids = _find_noisy_samples(config, sample_list, delta=delta, lr=lr)
    
    log_info(logger, f"HC: {len(hc_sample_ids)}, MC: {len(mc_sample_ids)}, LC: {len(lc_sample_ids)}")
    
    if config.save_ensembles_to_disk:
        # Saving noise_indices for analysis...
        np.save(os.path.join(config.logging_dir, "hc_sample_ids_" + str(config.noise_level) + ".npy"), hc_sample_ids)
        np.save(os.path.join(config.logging_dir, "mc_sample_ids_" + str(config.noise_level) + ".npy"), mc_sample_ids)
        np.save(os.path.join(config.logging_dir, "lc_sample_ids_" + str(config.noise_level) + ".npy"), lc_sample_ids)
    
    ####### for mc_set or lc_set, update the label to llm_empathy

    # Convert indices to sets for quick lookup
    mc_set = set(mc_sample_ids)
    lc_set = set(lc_sample_ids)
    
    # If sample_id is in mc_set or lc_set, update the label to llm_empathy
    for batch in train_dl:
        for i, sample_id in enumerate(batch["sample_id"]):
            sample_id = sample_id.item() # index tensor to scalar
            if sample_id in mc_set or sample_id in lc_set:
                batch["labels"][i] = batch[config.llm_column][i]

    log_info(logger, f"Updated labels for {len(mc_set) + len(lc_set)} samples")

    if config.save_ensembles_to_disk:
        # save updated train_dl
        torch.save(train_dl, os.path.join(config.logging_dir, "updated_train_dl.pt"))
        log_info(logger, f"Saved updated train_dl to {config.logging_dir}/updated_train_dl.pt")

    return train_dl
