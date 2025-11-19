from pathlib import Path
import numpy as np
import scipy

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

def plot_noise_removal_demo(outputs_path: Path, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    
    labels, preds, uncs, noise = _load_outputs(outputs_path)

    noisy_mask = noise > 0
    clean_mask = noise == 0

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True, gridspec_kw={"wspace": 0})
    # noise is binary now
    ax = axes[0]
    _ = ax.scatter(labels[clean_mask], preds[clean_mask], color="tab:blue", alpha=0.4, label="Clean")
    ax.scatter(labels[noisy_mask], preds[noisy_mask], color="tab:red", alpha=0.4, label="Noisy")
    ax.plot([1, 7], [1, 7], "k--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend()

    ax = axes[1]
    sc_unc = ax.scatter(labels, preds, c=uncs, alpha=0.4, cmap="viridis")
    ax.plot([1, 7], [1, 7], "k--")
    ax.set_xlabel("True")
    # ax.set_ylabel("Predicted")
    fig.colorbar(sc_unc, ax=ax, label="Uncertainty")

    save_as = f"{save_path}/uplme-noise-removal-demo.pdf"
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()
    print(f"Saved figure as {save_as}") 

def plot_noise_analysis(outputs_path: Path, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)

    labels, preds, uncs, noise = _load_outputs(outputs_path)
    abs_err = np.abs(preds - labels)
    
    noisy_mask = noise > 0
    clean_mask = noise == 0
    uncs_noisy = uncs[noisy_mask]
    uncs_clean = uncs[clean_mask]
    print(f"Mean uncertainty for noisy samples: {np.mean(uncs_noisy)}")
    print(f"Mean uncertainty for clean samples: {np.mean(uncs_clean)}")
    
    _, axes = plt.subplots(
        1, 3, figsize=(8, 3),
        gridspec_kw={"width_ratios" :[1.5, 0.5, 1.6]}
    )
    ax = axes[0]
    bins = np.linspace(np.min(uncs), np.max(uncs), 50)
    _ = ax.hist([uncs_noisy, uncs_clean], bins=bins,
                        stacked=False, color=["tab:red", "tab:blue"],
                        label=["Noisy", "Clean"])
    for group in [uncs_noisy, uncs_clean]:
        kde = scipy.stats.gaussian_kde(group)
        ax.plot(bins, kde(bins) * len(group) * (bins[1]-bins[0]), color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Frequency")

    ax = axes[1]
    ax.boxplot([uncs_noisy, uncs_clean], tick_labels=["Noisy", "Clean"])
    ax.set_ylabel("Uncertainty")
    ax.set_xlabel("Sample Type")

    ax = axes[2]
    ax.scatter(uncs[clean_mask], abs_err[clean_mask],
               color="tab:blue", alpha=0.6, label="Clean")
    ax.scatter(uncs[noisy_mask], abs_err[noisy_mask],
               color="tab:red", alpha=0.6, label="Noisy")
    ax.set_xlabel("Predictive Uncertainty")
    ax.set_ylabel("Absolute Prediction Error")

    rho, pval = scipy.stats.spearmanr(uncs, abs_err)
    print(f"Spearman's correlation coefficient: {rho}, p-value: {pval}")
    ax.text(
         0.95, 0.05, f"Spearman's $\\rho ={rho:.2f}$\np-value $={pval:.1e}$",
         transform=ax.transAxes, ha="right",
         bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]  # put "Noisy" first, then "Clean"
    ax.legend([handles[i] for i in order],
            [labels[i] for i in order])

    save_as = f"{save_path}/uplme-noise-removal-analysis.pdf"
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()
    print(f"Saved figure as {save_as}")

if __name__ == "__main__":
    output_file = "outputs/20250202_233546_single-prob_(2024,2022)-error-weighted-penalty-tuned/lr_3e-05_bs_16/seed_0/output_unc.npy"
    plot_noise_removal_demo(output_file, save_path=Path("outputs/noise-removal-demo"))
    plot_noise_analysis(output_file, save_path=Path("outputs/noise-removal-demo"))
