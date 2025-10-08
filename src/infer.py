import torch
from transformers import AutoTokenizer
from paired_texts_modelling import LitPairedTextModel

_device = None
_model = None
_tokeniser = None

def load_model(ckpt_path: str):
    global _model, _tokeniser, _device
    plm_name = "roberta-base"
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = LitPairedTextModel.load_from_checkpoint(ckpt_path).to(_device).eval()
    _tokeniser = AutoTokenizer.from_pretrained(
        plm_name,
        use_fast=True,
        add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
    )

@torch.inference_mode()
def predict(essay: str, article: str) -> tuple[float, float]:
    max_length = 512
    toks = _tokeniser(
        essay,
        article,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(_device)
    mean, var, _ = _model(toks)
    return mean.item(), var.item()

if __name__ == "__main__":
    ckpt_path = "outputs/2025-07-29/15-38-45_newsemp_cross-prob_lambdas-[1, 17.000527783218715, 49.309166285421874]_tune-False_single-model_4-passes-best-lambdas/seed_0/NoisEmpathy/awpc9zre/checkpoints/epoch=7-step=3000.ckpt"
    load_model(ckpt_path)
    # essay = "My heart just breaks for the people who lost their jobs."
    essay = "This doesn't sound too worrisome to me."
    article = "Getting a new job can bring a lot of joy and excitement."
    print(predict(essay, article))
    