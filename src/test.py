import os
import transformers
import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
import logging

from utils import resolve_logging_dir, log_info, resolve_seed_wise_checkpoint, process_seedwise_metrics
from model import LightningPLM, LightningProbabilisticPLMSingle, LightningProbabilisticPLMEnsemble

logger = logging.getLogger(__name__)

def test_plm(config: OmegaConf, have_label: bool, delta: float, seed: int, approach: str) -> dict:
    assert os.path.exists(config.test_from_checkpoint), "valid test_from_checkpoint is required for test_mode"
    
    datamodule = DataModuleFromRaw(
        delta=delta,
        seed=seed
    )
    trainer = L.Trainer(
        logger=False,
        devices=1,
        max_epochs=1
    )

    with trainer.init_module(empty_init=True):
        if approach == "single-probabilistic":
            model = LightningProbabilisticPLMSingle.load_from_checkpoint(config.test_from_checkpoint)
        elif approach == "ensemble-probabilistic":
            model = LightningProbabilisticPLMEnsemble.load_from_checkpoint(config.test_from_checkpoint)
        else:
            model = LightningPLM.load_from_checkpoint(config.test_from_checkpoint, config=config)

    log_info(logger, f"Loaded model from {config.test_from_checkpoint}")
    
    test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list, have_label=have_label)

    trainer.test(model=model, dataloaders=test_dl, verbose=True)

    if have_label:
        # metrics calculation is possibel only if we have labels
        metrics = {
            "test_pcc": trainer.callback_metrics["test_pcc"].item(),
            "test_ccc": trainer.callback_metrics["test_ccc"].item(),
            "test_rmse": trainer.callback_metrics["test_rmse"].item()
        }
    else:
        metrics = {}

    return metrics

def _test_multi_seeds(
        ckpt_parent_dir: str, config: OmegaConf, have_label: bool, delta: float,
        remove_noise: bool
    ) -> None:
    results = []

    for seed in config.seeds:
        config.seed = seed
        log_info(logger, f"Current seed: {config.seed}")
        config.test_from_checkpoint = resolve_seed_wise_checkpoint(ckpt_parent_dir, seed)
        config.logging_dir = ckpt_parent_dir
        test_metrics = test_plm(config, have_label, delta=delta, seed=seed, remove_noise=remove_noise)
        
        if have_label:
            # then we have metrics
            test_metrics["seed"] = seed
            log_info(logger, f"Metrics: {test_metrics}")
            results.append(test_metrics)

    if have_label:
        save_as = os.path.join(ckpt_parent_dir, "results_test.csv")
        process_seedwise_metrics(results, save_as)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    
    config_test = OmegaConf.load("config/config_test.yaml")
    
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_test)
    
    raise NotImplementedError("remove_noise is removed from the other parts of the code; use approach instead")

    if "test_from_checkpoint" in config:
        log_info(logger, f"Doing a single test using {config.test_from_checkpoint}")
        log_info(logger, f"Normal testing on {config.test_file_list}")
        config.logging_dir = resolve_logging_dir(config) # update customised logging_dir
        test_plm(config, config.have_label, delta=config.delta, seed=config.seeds[0])
    elif "test_from_ckpts_parent_dir" in config:
        log_info(logger, f"Multi-seed testing from {config.test_from_ckpts_parent_dir}")
        _test_multi_seeds(config.test_from_ckpts_parent_dir, config, have_label=config.have_label, delta=config.delta)
    else:
        raise ValueError("Either test_from_checkpoint or test_from_ckpts_parent_dir is required for testing")