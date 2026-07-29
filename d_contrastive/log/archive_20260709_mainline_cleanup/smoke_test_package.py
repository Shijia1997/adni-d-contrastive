#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd

from minimal_v0_contrastive import load_features_and_labels


ROOT = Path("/dcs07/zwang/data/adni_d")
FEATURES_DIR = ROOT / "data/embeddings_128_05152016"
D_CSV = ROOT / "data/master_smri_05152016/D_with_image_paths_full.csv"


def check_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"OK exists: {path}")


def main():
    required = [
        FEATURES_DIR / "swin_latent.npy",
        FEATURES_DIR / "image_id_order.npy",
        FEATURES_DIR / "swin_combat_train.npy",
        FEATURES_DIR / "swin_combat_test.npy",
        FEATURES_DIR / "swin_combat_train_ids.npy",
        FEATURES_DIR / "swin_combat_test_ids.npy",
        D_CSV,
        ROOT / "d_contrastive/AD_contrastive_key_results_20260605.csv",
    ]
    for path in required:
        check_exists(path)

    raw = np.load(FEATURES_DIR / "swin_latent.npy")
    image_ids = np.load(FEATURES_DIR / "image_id_order.npy", allow_pickle=True)
    train = np.load(FEATURES_DIR / "swin_combat_train.npy")
    test = np.load(FEATURES_DIR / "swin_combat_test.npy")
    print(f"raw latent shape: {raw.shape}")
    print(f"image_id_order: {len(image_ids)}")
    print(f"combat train/test: {train.shape} / {test.shape}")
    assert raw.shape[0] == len(image_ids)
    assert raw.shape[1] == 768
    assert train.shape[1] == test.shape[1] == 768

    meta = pd.read_csv(D_CSV, nrows=5)
    print(f"metadata columns sample: {list(meta.columns[:12])}")
    for col in ["RID", "d_mod3", "dx"]:
        assert col in meta.columns, f"missing column {col}"
    assert ("Image Data ID" in meta.columns) or ("Image_Data_ID" in meta.columns)

    # Full loader check for both raw and combat paths.
    for version in ["raw", "combat"]:
        data = load_features_and_labels(FEATURES_DIR, D_CSV, version=version)
        print(
            f"{version}: train {data['train_features'].shape}, "
            f"test {data['test_features'].shape}, "
            f"train RIDs {data['train_meta']['RID'].nunique()}, "
            f"test RIDs {data['test_meta']['RID'].nunique()}"
        )
        assert data["train_features"].shape[1] == 768
        assert data["test_features"].shape[1] == 768
        assert data["train_meta"]["d_mod3"].notna().all()
        assert data["test_meta"]["d_mod3"].notna().all()

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
