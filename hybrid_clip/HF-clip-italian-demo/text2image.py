import io
import os
import requests
import zipfile
import natsort
import gc
from PIL import Image
from PIL import UnidentifiedImageError

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from stqdm import stqdm
import streamlit as st
from jax import numpy as jnp
import transformers
from transformers import AutoTokenizer
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from modeling_hybrid_clip import FlaxHybridCLIP

import utils

@st.cache_resource
def get_model():
    return FlaxHybridCLIP.from_pretrained(
        "clip-italian/clip-italian"
        # `resume_download` is deprecated and will be removed in version 1.0.0.
        # Downloads always resume when possible.
        # If you want to force a new download, use `force_download=True`.
        , resume_download=None
    )

@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-italian-xxl-uncased", cache_dir="./", use_fast=True
        # `resume_download` is deprecated and will be removed in version 1.0.0.
        # Downloads always resume when possible.
        # If you want to force a new download, use `force_download=True`.
        , resume_download=None
    )


@st.cache_data
def download_images():
    # from sentence_transformers import SentenceTransformer, util
    img_folder = "photos/"
    if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
        os.makedirs(img_folder, exist_ok=True)

        photo_filename = "unsplash-25k-photos.zip"
        if not os.path.exists(photo_filename):  # Download dataset if does not exist
            print(f"Downloading {photo_filename}...")
            response = requests.get(
                f"http://sbert.net/datasets/{photo_filename}", stream=True
            )
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kb
            progress_bar = stqdm(
                total=total_size_in_bytes
            )  # , unit='iB', unit_scale=True
            content = io.BytesIO()
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                content.write(data)
            progress_bar.close()
            z = zipfile.ZipFile(content)
            # content.close()
            print("Extracting the dataset...")
            z.extractall(path=img_folder)
    print("Done.")


@st.cache_data
def get_image_features(dataset_name):
    if dataset_name == "Unsplash":
        return jnp.load("static/features/features.npy")
    else:
        return jnp.load("static/features/CC_embeddings.npy")


@st.cache_data
def load_urls(dataset_name):
    if dataset_name == "CC":
        with open("static/CC_urls.txt") as fp:
            urls = [l.strip() for l in fp.readlines()]
        return urls
    else:
        ValueError(f"{dataset_name} not supported here")


def get_image_transform(image_size):
    return Compose(
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


headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}


def app():

    st.title("From Text to Image")
    st.markdown(
        """
    
        ### 👋 Ciao!

        Here you can search for ~150.000 images in the Conceptual Captions dataset (CC) or in the Unsplash 25.000 Photos dataset.
        Even though we did not train on any of these images you will see most queries make sense. When you see errors, there might be two possibilities: 
        the model is answering in a wrong way or the image you are looking for is not in the dataset and the model is giving you the best answer it can get.
        
        
        
        🤌 Italian mode on! 🤌

        You can choose from one of the following examples:
        """
    )

    suggestions = [
        "Un gatto",
        "Due gatti",
        "Un fiore giallo",
        "Un fiore blu",
        "Una coppia in montagna",
        "Una coppia al tramonto",
    ]
    sugg_idx = -1

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button(suggestions[0]):
            sugg_idx = 0
    with col2:
        if st.button(suggestions[1]):
            sugg_idx = 1
    with col3:
        if st.button(suggestions[2]):
            sugg_idx = 2
    with col4:
        if st.button(suggestions[3]):
            sugg_idx = 3
    with col5:
        if st.button(suggestions[4]):
            sugg_idx = 4
    with col6:
        if st.button(suggestions[5]):
            sugg_idx = 5

    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        query = st.text_input("... or insert an Italian query text")
    with col2:
        dataset_name = st.selectbox("IR dataset", ["CC", "Unsplash"])

    query = suggestions[sugg_idx] if sugg_idx > -1 else query if query else ""

    if query:
        with st.spinner("Computing..."):

            if dataset_name == "Unsplash":
                download_images()

            image_features = get_image_features(dataset_name)
            model = get_model()
            tokenizer = get_tokenizer()

            if dataset_name == "Unsplash":
                image_size = model.config.vision_config.image_size
                dataset = utils.CustomDataSet(
                    "photos/", transform=get_image_transform(image_size)
                )
            elif dataset_name == "CC":
                dataset = load_urls(dataset_name)
            else:
                raise ValueError()

            N = 3

            image_paths = utils.find_image(
                query, model, dataset, tokenizer, image_features, N, dataset_name
            )

        for i, image_url in enumerate(image_paths):
            try:
                if dataset_name == "Unsplash":
                    st.image(image_url)
                elif dataset_name == "CC":
                    image_raw = requests.get(
                        image_url, stream=True, allow_redirects=True, headers=headers
                    ).raw
                    image = Image.open(image_raw).convert("RGB")
                    st.image(image, use_column_width=True)
                break
            except (UnidentifiedImageError) as e:
                if i == N - 1:
                    st.text(
                        f"Tried to show {N} different image URLS but none of them were reachabele.\nMaybe try a different query?"
                    )

        gc.collect()

        sugg_idx = -1
