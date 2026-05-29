"""Admin script: download datasets once and save them to a shared filesystem.

Problem: HuggingFace's cache uses file locks that are tied to the user who
created them.  On NFS/SLURM clusters this prevents other users from reading
the same cached dataset.

Solution: use dataset.save_to_disk() which writes plain Arrow files (no locks).
Other users then call load_from_disk() which reads directly — no locking needed.

Usage (run as an admin or any user with write access to SHARED_PATH):
    python prepare_datasets.py --shared_path /your/shared/path

After this script finishes, set in TrainConfig or via CLI:
    --dataset_local_path /your/shared/path/the_cauldron/ai2d   (for Cauldron)
    --dataset_local_path /your/shared/path/flickr30k           (for Flickr)

Then make the directory world-readable:
    chmod -R a+rX /your/shared/path
"""

import argparse
import os


CAULDRON_SUBSETS = ["ai2d", "vqa_v2", "llava"]   # safe subsets (no tiling needed)
FLICKR_REPO      = "AnyModal/flickr30k"
CAULDRON_REPO    = "HuggingFaceM4/the_cauldron"


def save_cauldron(shared_path: str, subsets: list[str]):
    from datasets import load_dataset

    for subset in subsets:
        out_dir = os.path.join(shared_path, "the_cauldron", subset)
        if os.path.exists(os.path.join(out_dir, "dataset_info.json")):
            print(f"  [skip] {subset} already exists at {out_dir}")
            continue
        print(f"  Downloading the_cauldron/{subset} …")
        ds = load_dataset(CAULDRON_REPO, subset)
        ds.save_to_disk(out_dir)
        print(f"  Saved to {out_dir}")


def save_flickr(shared_path: str):
    from datasets import load_dataset

    out_dir = os.path.join(shared_path, "flickr30k")
    if os.path.exists(os.path.join(out_dir, "dataset_info.json")):
        print(f"  [skip] flickr30k already exists at {out_dir}")
        return
    print("  Downloading flickr30k …")
    ds = load_dataset(FLICKR_REPO)
    ds.save_to_disk(out_dir)
    print(f"  Saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shared_path", type=str, required=True,
                   help="Shared directory where datasets will be saved.")
    p.add_argument("--skip_cauldron", action="store_true")
    p.add_argument("--skip_flickr", action="store_true")
    p.add_argument("--cauldron_subsets", type=str, default=",".join(CAULDRON_SUBSETS),
                   help="Comma-separated list of Cauldron subsets to download.")
    args = p.parse_args()

    os.makedirs(args.shared_path, exist_ok=True)

    if not args.skip_cauldron:
        print("=== The Cauldron ===")
        save_cauldron(args.shared_path, args.cauldron_subsets.split(","))

    if not args.skip_flickr:
        print("=== Flickr30k ===")
        save_flickr(args.shared_path)

    print(f"\nDone. Make the directory readable by all users:")
    print(f"    chmod -R a+rX {args.shared_path}")
    print(f"\nThen train with:")
    print(f"    python train.py --dataset_local_path {args.shared_path}/the_cauldron/ai2d")
    print(f"  or")
    print(f"    python train.py --dataset_type flickr --dataset_local_path {args.shared_path}/flickr30k")


if __name__ == "__main__":
    main()
