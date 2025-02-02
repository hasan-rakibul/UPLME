import logging
import torch
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
import lightning as L
import os
import glob

from utils import log_info
from model import LitBasicPLM, LitProbabilisticPLMSingle, LitProbabilisticPLMEnsemble
from preprocess import DataModuleFromRaw

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
            self,
            log_dir: str,
            data_path: list[str], 
            approach: str,
            delta: float = None,
            seed: int = 0,
            add_noise: bool = False,
            run_scratch: bool = True
        ):
        self.log_dir = log_dir
        self.approach = approach
        self.seed = seed

        self.run_scratch = run_scratch
        if self.run_scratch:
            dm = DataModuleFromRaw(
                delta=delta,
                seed=self.seed
            )
            
            self.dl = dm.get_test_dl(data_path_list=data_path, batch_size=16, have_label=True, add_noise=add_noise)

    def test(self, model_path: str, save_uc_metrics: bool = False):
        trainer = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        with trainer.init_module(empty_init=True):
            if self.approach == "basic":
                model = LitBasicPLM.load_from_checkpoint(model_path)
            elif self.approach == "single-prob":
                model = LitProbabilisticPLMSingle.load_from_checkpoint(model_path, save_uc_metrics=save_uc_metrics)
            elif self.approach == "ensemble-prob":
                model = LitProbabilisticPLMEnsemble.load_from_checkpoint(model_path, save_uc_metrics=save_uc_metrics)
            else:
                raise ValueError(f"Invalid approach: {self.approach}")

        trainer.test(model=model, dataloaders=self.dl, verbose=True)

        try:
            metrics = {
                "test_pcc": trainer.callback_metrics["test_pcc"].item(),
                "test_ccc": trainer.callback_metrics["test_ccc"].item(),
                "test_rmse": trainer.callback_metrics["test_rmse"].item()
            }
        except KeyError:
            metrics = {}
        
        return metrics

    def retrieve_uncertainty_metrics_scratch(self):
        model = LitProbabilisticPLMSingle.load_from_checkpoint(self.model_path)
        
        dm = DataModuleFromRaw(
            delta=2.0,
            seed=0,
            tokeniser_plm="roberta-base"
        )
        dl = dm.get_train_dl(data_path_list=self.data_path, batch_size=16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        uncs = []
        labels = []
        preds = []
        noise = []

        model.eval()
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                mean, var = model(batch)
                unc = torch.sqrt(var).cpu().numpy()
                uncs.extend(unc)
                labels.extend(batch["labels"].cpu().numpy())
                preds.extend(mean.cpu().numpy())
                noise.extend(batch["noise"].cpu().numpy())

        return np.array(uncs), np.array(labels), np.array(preds), np.array(noise)
    
    def plot_uncertainty(self, model_path: str = None, save_uc_metrics: bool = False):
        if self.run_scratch:
            _ = self.test(model_path=model_path, save_uc_metrics=save_uc_metrics)

        # if load_npy:
        #     uncs = np.load(f"{self.log_dir}/uncs.npy")
        #     labels = np.load(f"{self.log_dir}/labels.npy")
        #     preds = np.load(f"{self.log_dir}/preds.npy")
        #     noise = np.load(f"{self.log_dir}/noise.npy")
        # else:
        #     uncs, labels, preds, noise = self.retrieve_uncertainty_metrics_scratch()
        #     np.save(f"{self.log_dir}/uncs.npy", uncs)
        #     np.save(f"{self.log_dir}/labels.npy", labels)
        #     np.save(f"{self.log_dir}/preds.npy", preds)
        #     np.save(f"{self.log_dir}/noise.npy", noise)


        unc_file = glob.glob(os.path.join(self.log_dir, "**/*.npy"), recursive=True)
        assert len(unc_file) == 1, f"Multiple or no npy files found in {self.log_dir}"
        unc_file = unc_file[0]
        outputs = np.load(unc_file, allow_pickle=True)

        var = outputs.item().get("var")
        uncs = np.sqrt(var)
        labels = outputs.item().get("labels")
        preds = outputs.item().get("mean")
        noise = outputs.item().get("noise")

        noisy_mask = noise > 0
        uncs_noisy = uncs[noisy_mask]
        uncs_clean = uncs[~noisy_mask]
        log_info(logger, f"Mean uncertainty for noisy samples: {np.mean(uncs_noisy)}")
        log_info(logger, f"Mean uncertainty for clean samples: {np.mean(uncs_clean)}")

        _, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].scatter(labels, preds, c=uncs, cmap="coolwarm")
        ax[0, 0].plot([1, 7], [1, 7], "k--")
        ax[0, 0].set_xlabel("True")
        ax[0, 0].set_ylabel("Predicted")
        ax[0, 1].scatter(labels, preds, c=noise, cmap="coolwarm")
        ax[0, 1].plot([1, 7], [1, 7], "k--")
        ax[0, 1].set_xlabel("True")
        ax[0, 1].set_ylabel("Predicted")

        ax[1, 0].hist(uncs_noisy, bins=50, alpha=0.5, label="Noisy", color="red")
        ax[1, 0].hist(uncs_clean, bins=50, alpha=0.5, label="Clean", color="blue")
        ax[1, 0].legend()
        ax[1, 0].set_xlabel("Uncertainty")
        ax[1, 0].set_ylabel("Frequency")

        ax[1, 1].boxplot([uncs_noisy, uncs_clean], tick_labels=["Noisy", "Clean"])
        ax[1, 1].set_ylabel("Uncertainty")
        ax[1, 1].set_xlabel("Sample type")

        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/uncertainty.pdf")
        log_info(logger, f"Saved plot at {self.log_dir}/uncertainty.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, help="Path to the log directory")
    parser.add_argument("-r", "--run_scratch", action="store_true", help="Run the evaluation from scratch")

    args = parser.parse_args()

    approach = "single-prob"
    
    if args.run_scratch:
        model_path = glob.glob(os.path.join(args.log_dir, "**/*.ckpt"), recursive=True)
        assert len(model_path) == 1, f"Multiple or no ckpt files found in {args.log_dir}"
        model_path = model_path[0]
    else:
        model_path = None

    constants = OmegaConf.load("config/config_common.yaml")
    data_path_numeric = [2024, 2022]
    data_path = []
    for data in data_path_numeric:
        data_path.append(getattr(constants[data], "train_llama"))

    vu = ModelEvaluator(
        log_dir=args.log_dir,
        data_path=data_path,
        approach=approach,
        delta=constants.delta,
        add_noise=True,
        run_scratch=args.run_scratch
    )
    vu.plot_uncertainty(model_path=model_path, save_uc_metrics=True)
