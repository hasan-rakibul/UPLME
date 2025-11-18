from pathlib import Path
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

import matplotlib.pyplot as plt
import scienceplots

# load latex in Setonix: module load psc-texlive-2025
plt.style.use({"science", "ieee", "tableau-colorblind10"})
# plt.style.use({"science", "ieee", "tableau-colorblind10", "no-latex"})
    
from preprocess import PairedTextDataModule
from paired_texts_modelling import LitPairedTextModel
from consistency_modelling import LitUCVME

def main():
    ckpt_paths = [
        # "outputs/2025-11-15/12-41-32_newsemp_cross-basic_lambdas-[None, None, 0]_tune-False_baseline/seed_0/NoisEmpathy/f4j67cge/checkpoints/epoch=8-step=3400.ckpt",
        # "outputs/2025-07-30/16-35-59_newsemp_cross-prob_lambdas-[1, None, None, None, None]_tune-False_4-passes-ucvme/seed_0/NoisEmpathy/kaaqmzi4/checkpoints/epoch=7-step=2800.ckpt",
        # "outputs/2025-07-28/14-22-44_newsemp_cross-prob_lambdas-[1, 9.110462266012783, 5.5635098435909764]_tune-False_single-model_4-passes-best/seed_0/NoisEmpathy/a3ft2a6y/checkpoints/epoch=7-step=3000.ckpt",
        "outputs/2025-11-08/10-13-04_newsemp_cross-prob_lambdas-[1, 17.3, 0.3]_tune-False_single-model_4-passes_ablation-lambdas/seed_0/NoisEmpathy/6mc7c1xc/checkpoints/epoch=6-step=2400.ckpt"
    ]

    identifiers = [
        # "PLM-MLP",
        # "UCVME",
        "UPLME",
    ]

    plot_tsne_umap(
        ckpt_paths=ckpt_paths,
        identifiers=identifiers,
        save_dir="outputs/tsne_umap",
        seed=0,
        use_cache=True,
        force_recompute=False,
        n_components=3
    )

def plot_tsne_umap(
    ckpt_paths: list[str],
    identifiers: list[str],
    save_dir: str,
    seed: int,
    use_cache: bool = True,
    force_recompute: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 3
) -> None:
    results = []
    for ckpt, identifier in zip(ckpt_paths, identifiers):
        print(f"Working on {identifier}: {ckpt}")
        embeddings, tsne_embeddings, labels = _get_or_create_tsne_data(
            identifier=identifier,
            ckpt_path=ckpt,
            seed=seed,
            save_dir=save_dir,
            use_cache=use_cache,
            force_recompute=force_recompute,
            n_components=n_components
        )
        _, umap_embeddings, _ = _get_or_create_umap_data(
            identifier=identifier,
            ckpt_path=ckpt,
            seed=seed,
            save_dir=save_dir,
            use_cache=use_cache,
            force_recompute=force_recompute,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components
        )
        results.append((embeddings, tsne_embeddings, umap_embeddings, labels))

    num_plot_pairs = len(results)
    if num_plot_pairs == 0:
        raise ValueError("No embeddings available for plotting.")

    side_by_side = num_plot_pairs == 1
    fig_width = 10 if side_by_side else 5 * num_plot_pairs
    fig_height = 5 if side_by_side else 10
    fig = plt.figure(figsize=(fig_width, fig_height))
    # fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    scatter_ref = None

    cmap = plt.colormaps["viridis_r"]
    norm = plt.Normalize(1, 7)

    for idx, (embeddings, tsne_embeddings, umap_embeddings, labels) in enumerate(results):
        if tsne_embeddings.shape[1] < 2:
            raise ValueError("t-SNE embeddings must have at least two components for plotting.")
        if umap_embeddings.shape[1] < 2:
            raise ValueError("UMAP embeddings must have at least two components for plotting.")

        tsne_proj = "3d" if tsne_embeddings.shape[1] >= 3 else None
        umap_proj = "3d" if umap_embeddings.shape[1] >= 3 else None

        if side_by_side:
            tsne_ax = fig.add_subplot(1, 2, 1, projection=tsne_proj)
            umap_ax = fig.add_subplot(1, 2, 2, projection=umap_proj)
        else:
            tsne_ax = fig.add_subplot(2, num_plot_pairs, idx + 1, projection=tsne_proj)
            umap_ax = fig.add_subplot(2, num_plot_pairs, num_plot_pairs + idx + 1, projection=umap_proj)

        if tsne_proj:
            tsne_scatter = tsne_ax.scatter(
                tsne_embeddings[:, 0],
                tsne_embeddings[:, 1],
                tsne_embeddings[:, 2],
                c=labels,
                cmap=cmap,
                norm=norm
            )
            tsne_ax.set_zlabel("t-SNE 3")
        else:
            tsne_scatter = tsne_ax.scatter(
                tsne_embeddings[:, 0],
                tsne_embeddings[:, 1],
                c=labels,
                cmap=cmap,
                norm=norm
            )
        scatter_ref = scatter_ref or tsne_scatter
        tsne_ax.set_xlabel("t-SNE 1")
        tsne_ax.set_ylabel("t-SNE 2")

        if umap_proj:
            umap_ax.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                umap_embeddings[:, 2],
                c=labels,
                cmap=cmap,
                norm=norm
            )
            umap_ax.set_zlabel("UMAP 3")
        else:
            umap_ax.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                c=labels,
                cmap=cmap,
                norm=norm
            )
        umap_ax.set_xlabel("UMAP 1")
        umap_ax.set_ylabel("UMAP 2")

        # NOTE: skeptical about that the score signify in this case, also because we have continuous scale
        # score = _embedding_silhouette_score(embeddings, labels)
        # tsne_ax.set_title(identifiers[idx] + " (t-SNE)\n" + r"\textbf{Silhouette score: $\mathbf{" + f"{score}" + r"}$}")
        # umap_ax.set_title(identifiers[idx] + " (UMAP)\n" + r"\textbf{Silhouette score: $\mathbf{" + f"{score}" + r"}$}")

    fig.subplots_adjust(wspace=0.2)

    cbar = fig.colorbar(
        scatter_ref,
        ax=fig.axes,
        shrink=0.75 if side_by_side else 0.65,
        aspect=50,
        pad=0.07,
        # orientation="horizontal",
        location="right"
    )
    cbar.set_label("Empathy score")
    # cbar.ax.xaxis.set_label_position("top")
    # cbar.ax.xaxis.set_ticks_position("bottom")

    output_path = Path(save_dir) / "tsne_umap.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Save t-SNE/UMAP as {output_path}")

def _collect_embeddings_from_model(
    ckpt_path: str,
    dataloader,
    identifier: str
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if identifier in ["PLM-MLP", "UPLME"]:
        model = LitPairedTextModel.load_from_checkpoint(ckpt_path)
    elif identifier in ["UCVME"]:
        model = LitUCVME.load_from_checkpoint(ckpt_path)
    
    model.eval()
    model.on_test_start() # enable MCD
    model.to(device)

    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            _, _, _, sentence_representation = model.forward(batch_on_device)
            embeddings.append(sentence_representation.detach().cpu())

            if "labels" in batch_on_device:
                labels.append(batch_on_device["labels"].detach().cpu())

    if not embeddings:
        raise ValueError("No embeddings were collected. Check that the dataloader yields data.")

    embeddings_tensor = torch.cat(embeddings, dim=0)
    labels_tensor = torch.cat(labels, dim=0) if labels else torch.empty(0)

    return embeddings_tensor.numpy(), labels_tensor.numpy()

def _get_or_create_embeddings(
    identifier: str,
    save_dir: Path | str,
    seed: int,
    ckpt_path: str,
    use_cache: bool,
    force_recompute: bool
) -> tuple[np.ndarray, np.ndarray, Path]:
    if identifier == "PLM-MLP":
        approach = "cross-basic" 
        plm_names = ["roberta-base"]
    elif identifier == "UPLME":
        approach = "cross-prob"
        plm_names = ["roberta-base"]
    elif identifier == "UCVME":
        approach = "cross-prob"
        plm_names = ["roberta-base", "roberta-base"]

    train_files = [
        'data/NewsEmp2024/trac3_EMP_train_llama.tsv',
        'data/NewsEmp2022/messages_train_ready_for_WS_llama.tsv',
        'data/NewsEmp2022/messages_dev_features_ready_for_WS_2022_llama.tsv'
    ]

    test_files = [
        "data/NewsEmp2024/test_data_with_labels/goldstandard_EMP.csv"
    ]

    data_paths = train_files
    
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    emb_path = save_dir / f"embeddings_{identifier}.npy"
    labels_path = save_dir / f"labels_{identifier}.npy"

    if (
        use_cache
        and not force_recompute
        and emb_path.exists()
        and labels_path.exists()
    ):
        embeddings = np.load(emb_path)
        labels = np.load(labels_path)
        labels = labels.squeeze() if labels.size else np.array([])
        return embeddings, labels, save_dir

    dm = PairedTextDataModule(
        noise_level=0.0,
        delta=None,
        tokeniser_plms=plm_names,
        tokenise_paired_texts_each_tokeniser=True \
            if approach in ["cross-basic", "cross-prob"] else False
    )

    train_dl = dm.get_train_dl(
        data_path_list=data_paths,
        batch_size=16,
        sanitise_newsemp_labels=False,
        add_noise=False,
        seed=seed,
        is_newsemp=True,
        do_augment=True,
        lbl_split=1.0
    )

    embeddings, labels = _collect_embeddings_from_model(
        ckpt_path=ckpt_path,
        dataloader=train_dl,
        identifier=identifier
    )

    if embeddings.shape[0] < 2:
        raise ValueError("t-SNE requires at least two samples to compute embeddings.")

    if labels.size:
        labels = labels.squeeze()

    np.save(emb_path, embeddings)
    np.save(labels_path, labels if labels.size else np.array([]))

    return embeddings, labels, save_dir

def _get_or_create_tsne_data(
    identifier: str,
    save_dir: Path | str,
    seed: int,
    ckpt_path: str,
    use_cache: bool,
    force_recompute: bool,
    n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels, save_dir = _get_or_create_embeddings(
        identifier=identifier,
        save_dir=save_dir,
        seed=seed,
        ckpt_path=ckpt_path,
        use_cache=use_cache,
        force_recompute=force_recompute
    )

    tsne_path = Path(save_dir) / f"embeddings_{identifier}_tsne.npy"
    if (
        use_cache
        and not force_recompute
        and tsne_path.exists()
    ):
        tsne_embeddings = np.load(tsne_path)
    else:
        tsne = TSNE(
            n_components=n_components,
            perplexity=30,
            random_state=seed,
            method="exact",
            n_jobs=-1,
            verbose=1
        )
        tsne_embeddings = tsne.fit_transform(embeddings)
        np.save(tsne_path, tsne_embeddings)

    return embeddings, tsne_embeddings, labels

def _get_or_create_umap_data(
    identifier: str,
    save_dir: Path | str,
    seed: int,
    ckpt_path: str,
    use_cache: bool,
    force_recompute: bool,
    n_neighbors: int,
    min_dist: float,
    n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels, save_dir = _get_or_create_embeddings(
        identifier=identifier,
        save_dir=save_dir,
        seed=seed,
        ckpt_path=ckpt_path,
        use_cache=use_cache,
        force_recompute=force_recompute
    )

    umap_path = Path(save_dir) / f"embeddings_{identifier}_umap.npy"
    if (
        use_cache
        and not force_recompute
        and umap_path.exists()
    ):
        umap_embeddings = np.load(umap_path)
    else:
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed
        )
        umap_embeddings = umap_model.fit_transform(embeddings)
        np.save(umap_path, umap_embeddings)

    return embeddings, umap_embeddings, labels

def _embedding_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n_clusters = 6 # [1.0, 2.0), .[)
    # Convert continuous labels to discrete clusters by binning
    bins = np.linspace(1.0, 7 + 1e-6, num=n_clusters+1) # +1e-6 to avoid the last bin being only 7.0
    label_cluster = np.digitize(labels, bins)
    score = silhouette_score(embeddings, label_cluster)
    return round(score, 3)

if __name__ == "__main__":
    main()
