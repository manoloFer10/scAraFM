import sys
from pathlib import Path


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    repo_root = Path(__file__).resolve().parent.parent
    repo_id = "manufernandezbur/scAraFM-pretraining-finetuning"

    print(f"Downloading data and weights from {repo_id} ...")
    print(f"Destination: {repo_root}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(repo_root),
        allow_patterns=["data/**", "model/weights/**"],
        ignore_patterns=[".gitattributes", "README.md"],
    )

    print("Download complete:")
    print(f"  - data:    {repo_root / 'data'}")
    print(f"  - weights: {repo_root / 'model' / 'weights'}")


if __name__ == "__main__":
    main()
