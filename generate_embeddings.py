import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from gloria.builder import build_img_model, build_text_model, build_transformation
from gloria.datasets.data_module import pretraining_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=["base", "mimic"], required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.weights == "base":
        args.weights_path = "pretrained/chexpert_resnet50.ckpt"
    elif args.weights == "mimic":
        args.weights_path = (
            "data/ckpt/gloria_pretrain_1.0/2025_01_10_10_55_38/epoch=9-step=41549.ckpt"
        )
    else:
        raise NotImplementedError("Unknown value for weights: " + args.weights)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main(args):
    cfg = OmegaConf.load("configs/chexpert_pretrain_config.yaml")
    cfg.data.text.full_report = True

    # hard code for deterministic transforms
    transform = build_transformation(cfg, "test")
    ds = pretraining_dataset.MultimodalPretrainingDataset(
        cfg,
        split=args.split,
        transform=transform,
    )
    dl = DataLoader(
        ds,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=pretraining_dataset.multimodal_collate_fn,
    )

    img_encoder = build_img_model(cfg)
    text_encoder = build_text_model(cfg)

    sd = torch.load(args.weights_path, "cpu")["state_dict"]
    text_sd = {
        k.replace("gloria.text_encoder.", ""): v
        for k, v in sd.items()
        if k.startswith("gloria.text_encoder.")
    }
    img_sd = {
        k.replace("gloria.img_encoder.", ""): v
        for k, v in sd.items()
        if k.startswith("gloria.img_encoder.")
    }

    img_encoder.load_state_dict(img_sd)
    text_encoder.load_state_dict(text_sd)

    img_encoder = img_encoder.eval().to("cuda")
    text_encoder = text_encoder.eval().to("cuda")

    img_batches = []
    text_batches = []
    subject_ids = []
    study_ids = []
    dicom_ids = []
    for batch in tqdm(dl):
        for path in batch["path"]:
            parts = path.split("/")
            subject_ids.append(int(parts[-3][1:]))
            study_ids.append(int(parts[-2][1:]))
            dicom_ids.append(parts[-1][:-4])
        with torch.no_grad():
            h = img_encoder(batch["imgs"].to("cuda"))
            img_emb = img_encoder.global_embedder(h)
            _, text_emb, _ = text_encoder(
                batch["caption_ids"].to("cuda"),
                batch["attention_mask"].to("cuda"),
                batch["token_type_ids"].to("cuda"),
            )
        img_batches.append(img_emb.to("cpu"))
        text_batches.append(text_emb.to("cpu"))
    img_embs = torch.cat(img_batches, dim=0).numpy()
    text_embs = torch.cat(text_batches, dim=0).numpy()
    df = pd.DataFrame(
        {
            "subject_ids": subject_ids,
            "study_ids": study_ids,
            "dicom_ids": dicom_ids,
        }
    )
    np.save(
        os.path.join(
            args.output_dir, f"gloria-{args.weights}-{args.split}-split-img-embeds.npy"
        ),
        img_embs,
    )
    np.save(
        os.path.join(
            args.output_dir, f"gloria-{args.weights}-{args.split}-split-text-embeds.npy"
        ),
        text_embs,
    )
    df.to_csv(
        os.path.join(
            args.output_dir, f"gloria-{args.weights}-{args.split}-split-metadata.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
