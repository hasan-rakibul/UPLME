from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use({"science", "ieee", "tableau-colorblind10"})

def _load_outputs(outputs_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the .npy dictionary and return labels, predictions, uncertainty, and noise."""
    data = np.load(outputs_path, allow_pickle=True)
    data = data.item()

    for key in ("labels", "mean", "var", "noise"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in {outputs_path}")

    labels = data["labels"]
    preds = data["mean"]
    uncs = np.sqrt(data["var"])
    noise = data["noise"]
    return labels, preds, uncs, noise

def plot_penalty_loss(outputs_path: Path, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    
    labels, preds, uncs, noise = _load_outputs(outputs_path)

    noisy_mask = noise > 0
    uncs_noisy = uncs[noisy_mask]
    uncs_clean = uncs[~noisy_mask]
    print(f"Mean uncertainty for noisy samples: {np.mean(uncs_noisy)}")
    print(f"Mean uncertainty for clean samples: {np.mean(uncs_clean)}")

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
    plt.savefig(f"{save_path}/uplme-noise-removal-demo.pdf")
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    output_file = "outputs/20250202_233546_single-prob_(2024,2022)-error-weighted-penalty-tuned/lr_3e-05_bs_16/seed_0/output_unc.npy"
    plot_penalty_loss(output_file, save_path=Path("outputs/noise-removal-demo"))
