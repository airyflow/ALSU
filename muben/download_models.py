#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def setup_directories(base_dir):
    """Ensure all model directories exist."""
    paths = {
        "grover": base_dir / "grover",
        "unimol": base_dir / "unimol",
        "molformer": base_dir / "molformer"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def download_grover(target_dir):
    print("\n=========================================")
    print("1. DOWNLOADING GROVER PRETRAINED WEIGHTS")
    print("=========================================")
    try:
        import gdown
    except ImportError:
        print("Installing gdown via pip for Google Drive downloads...")
        subprocess.run(["pip", "install", "gdown"], check=True)
        import gdown

    # File target destinations
    output_base = target_dir / "grover_base.pt"
    output_large = target_dir / "grover_large.pt"

    # Isolated Google Drive IDs
    id_grover_base = "1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl"
    id_grover_large = "1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw"

    # FIX: Removed the invalid fuzzy=True parameter
    print("-> Downloading GROVER Base...")
    gdown.download(id=id_grover_base, output=str(output_base), quiet=False)

    print("\n-> Downloading GROVER Large...")
    gdown.download(id=id_grover_large, output=str(output_large), quiet=False)



def download_unimol(target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/dptech/Uni-Mol-Models/resolve/main"

    print("\n=========================================")
    print("DOWNLOADING UNI-MOL PRETRAINED MODELS")
    print("=========================================")

    UNI_MOL_FILES = [
    "mol_pre_all_h_220816.pt",
    "mol_pre_no_h_220816.pt",
    "mp_all_h_230313.pt",
    "oled_pre_no_h_230101.pt",
    "poc_pre_220816.pt",
    "pocket_pre_220816.pt",
    "mol.dict.txt",
    "mp.dict.txt",
    "oled.dict.txt",
    "poc.dict.txt",
    "dict_coarse.txt",
    ]

    for fname in UNI_MOL_FILES:
        url = f"{base_url}/{fname}"
        out_path = target_dir / fname
        print(f"-> Downloading {fname} ...")
        subprocess.run(["wget", url, "-O", str(out_path)], check=True)

    print("-> All Uni-Mol models downloaded successfully.")



def download_molformer(target_dir):
    print("\n=========================================")
    print("3. DOWNLOADING MOLFORMER 1D TRANSFORMER WEIGHTS")
    print("=========================================")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub via pip...")
        subprocess.run(["pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    print("-> Streaming MoLFormer-XL-both-10pct layers from HuggingFace Hub...")
    snapshot_download(
        repo_id="ibm-research/MoLFormer-XL-both-10pct",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    print("-> MoLFormer caching complete.")


if __name__ == "__main__":
    # Centralized Model Shared Folder Layout Target
    BASE_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
    
    print(f"Initializing Shared Model Zoo at: {BASE_MODEL_DIR}")
    dirs = setup_directories(BASE_MODEL_DIR)
    
    # Execute structural downloads
    download_grover(dirs["grover"])
    download_unimol(dirs["unimol"])
    download_molformer(dirs["molformer"])
    
    print("\n=========================================")
    print("ALL PRETRAINED CORE ARCHITECTURES DOWNLOADED SUCCESSFULLY!")
    print(f"Location: {BASE_MODEL_DIR}")
    print("=========================================")