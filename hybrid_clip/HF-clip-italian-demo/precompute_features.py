from tqdm import tqdm
from argparse import ArgumentParser
from jax import numpy as jnp
from torchvision import datasets, transforms
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from modeling_hybrid_clip import FlaxHybridCLIP
import utils
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("out_file")
    args = parser.parse_args()

    model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")

    tokenizer = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-italian-xxl-uncased", cache_dir=None, use_fast=True
    )

    image_size = model.config.vision_config.image_size

    val_preprocess = transforms.Compose(
        [
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    dataset = utils.CustomDataSet(args.in_dir, transform=val_preprocess)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        drop_last=False,
    )

    image_features = utils.precompute_image_features(model, loader)
    jnp.save(f"static/features/{args.out_file}", image_features)
