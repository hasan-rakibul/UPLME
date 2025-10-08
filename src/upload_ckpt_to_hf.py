from huggingface_hub import HfApi

def main():
    api = HfApi()

    api.upload_file(
        path_or_fileobj="outputs/2025-07-29/15-38-45_newsemp_cross-prob_lambdas-[1, 17.000527783218715, 49.309166285421874]_tune-False_single-model_4-passes-best-lambdas/seed_0/NoisEmpathy/awpc9zre/checkpoints/epoch=7-step=3000.ckpt",
        path_in_repo="UPLME_NewsEmp_tuned-lambdas.ckpt",
        repo_id="rhasan/UPLME",
        repo_type="model"
    )

if __name__ == "__main__":
    main()