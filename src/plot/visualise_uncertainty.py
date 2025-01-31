import logging
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import log_info
from model import LightningProbabilisticPLMSingle
from preprocess import DataModuleFromRaw

logger = logging.getLogger(__name__)

class VisualiseUncertainty:
    def __init__(self, model_path: str, data_path: list):
        self.log_dir = "log"
        self.model_path = model_path
        self.data_path = data_path

    def retrieve_uncertainty_metrics(self):
        model = LightningProbabilisticPLMSingle.load_from_checkpoint(self.model_path)
        
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
                batch = {k: v.to(self.device) for k, v in batch.items()}
                mean, var = model(batch)
                unc = torch.sqrt(var).cpu().numpy()
                uncs.extend(unc)
                labels.extend(batch["labels"].cpu().numpy())
                preds.extend(mean.cpu().numpy())
                noise.extend(batch["noise"].cpu().numpy())

        return np.array(uncs), np.array(labels), np.array(preds), np.array(noise)
    
    def plot_uncertainty(self, load_npy: bool = False):
        if load_npy:
            uncs = np.load(f"{self.log_dir}/uncs.npy")
            labels = np.load(f"{self.log_dir}/labels.npy")
            preds = np.load(f"{self.log_dir}/preds.npy")
            noise = np.load(f"{self.log_dir}/noise.npy")
        else:
            uncs, labels, preds, noise = self.retrieve_uncertainty_metrics()
            np.save(f"{self.log_dir}/uncs.npy", uncs)
            np.save(f"{self.log_dir}/labels.npy", labels)
            np.save(f"{self.log_dir}/preds.npy", preds)
            np.save(f"{self.log_dir}/noise.npy", noise)
        
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
    model_path = "log/20250131_084832_single-probabilistic_(2024,2022)/lr_3e-05_bs_16/seed_1234/NoisEmpathy/7iqnkxpx/checkpoints/epoch=8-step=1188.ckpt"

    constants = OmegaConf.load("config/config_common.yaml")
    data_path_numeric = [2024, 2022]
    data_path = []
    for data in data_path_numeric:
        data_path.append(getattr(constants[data], "train_llama"))

    vu = VisualiseUncertainty(model_path=model_path, data_path=data_path)
    vu.plot_uncertainty(load_npy=True)
