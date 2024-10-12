import os
import natsort
from tqdm import tqdm
import torch
from jax import numpy as jnp
from PIL import Image as PilImage


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def get_image_name(self, idx):
        return self.total_imgs[idx]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PilImage.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def text_encoder(text, model, tokenizer):
    inputs = tokenizer(
        [text],
        max_length=96,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    embedding = model.get_text_features(
        inputs["input_ids"], 
        inputs["attention_mask"])[0]
    norms = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
    embedding = embedding / norms
    return jnp.expand_dims(embedding, axis=0), norms


def image_encoder(image, model):
    image = image.permute(1, 2, 0).numpy()
    image = jnp.expand_dims(image, axis=0)  #  add batch size
    features = model.get_image_features(image,)
    norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
    features = features / norms
    return features, norms


def precompute_image_features(model, loader):
    image_features = []
    for i, (images) in enumerate(tqdm(loader)):
        images = images.permute(0, 2, 3, 1).numpy()
        features = model.get_image_features(images,)
        features /= jnp.linalg.norm(features, axis=-1, keepdims=True)
        image_features.extend(features)
    return jnp.array(image_features)


def find_image(text_query, model, dataset, tokenizer, image_features, n, dataset_name):
    zeroshot_weights, _ = text_encoder(text_query, model, tokenizer)
    distances = jnp.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n + 1):
        idx = jnp.argsort(distances, axis=0)[-i, 0]

        if dataset_name == "Unsplash":
            file_paths.append("photos/" + dataset.get_image_name(idx))
        elif dataset_name == "CC":
            file_paths.append(dataset[idx])
        else:
            raise ValueError(f"{dataset_name} not supported here")
    return file_paths
