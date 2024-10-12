import streamlit as st
from text2image import get_model, get_tokenizer, get_image_transform
from utils import text_encoder
from torchvision import transforms
from PIL import Image
from jax import numpy as jnp
import pandas as pd
import numpy as np
import requests
import psutil
import time
import jax
import gc


headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)


def resize_longer(image, longer_size=224):
    old_size = image.size
    ratio = float(longer_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    return image


def pad_to_square(image):
    (a,b)=image.shape[:2]
    if a<b:
        ah = (b - a) // 2
        padding=((ah,b - a -ah), (0,0), (0,0))
    else:
        bh = (a - b) // 2
        padding=((0,0), (bh,a-b-bh), (0,0))
    return np.pad(image, padding,mode='constant',constant_values=127)


def image_encoder(image, model):
    image = np.transpose(image, (0, 2, 3, 1))
    features = model.get_image_features(image)
    feature_norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
    features = features / feature_norms
    return features, feature_norms


def gen_image_batch(image_url, image_size=224, pixel_size=10):
    n_pixels = image_size // pixel_size + 1

    image_batch = []
    masks = []
    is_vertical = []
    is_horizontal = []
    
    image_raw = requests.get(image_url, stream=True).raw
    image = Image.open(image_raw).convert("RGB")
    image = np.array(resize_longer(image, longer_size=image_size))
    gray = np.ones_like(image) * 127
    mask = np.ones_like(image[:,:,:1])

    image_batch.append(image)
    masks.append(mask)
    is_vertical.append(True)
    is_horizontal.append(True)


    for i in range(0, image.shape[0] // pixel_size + 1):
        for j in range(i+1, image.shape[0] // pixel_size + 2):
            m = mask.copy()
            m[:min(i*pixel_size, image_size), :] = 0
            m[min(j*pixel_size, image_size):, :] = 0
            neg_m = 1 - m
            image_batch.append(image.copy() * m + gray * neg_m)
            masks.append(m)
            is_vertical.append(False)
            is_horizontal.append(True)

    for i in range(0, image.shape[1] // pixel_size + 1):
        for j in range(i+1, image.shape[1] // pixel_size + 2):
            m = mask.copy()
            m[:, :min(i*pixel_size, image_size)] = 0
            m[:, min(j*pixel_size, image_size):] = 0
            neg_m = 1 - m
            image_batch.append(image.copy() * m + gray * neg_m)
            masks.append(m)
            is_vertical.append(True)
            is_horizontal.append(False)

    return image_batch, masks, is_vertical, is_horizontal


def get_heatmap(image_url, text, pixel_size=10, iterations=3):
    tokenizer = get_tokenizer()
    model = get_model()
    image_size = model.config.vision_config.image_size

    images, masks, vertical, horizontal = gen_image_batch(image_url, pixel_size=pixel_size)
    input_image = images[0].copy()

    images = np.stack([preprocess(pad_to_square(image)) for image in images], axis=0)
    image_embeddings, embedding_norms = image_encoder(images, model)
    text_embeddings, _ = text_encoder(text, model, tokenizer)
    
    vertical_scores = jnp.zeros((masks[0].shape[1], 512))
    vertical_masks = jnp.zeros((masks[0].shape[1], 1))
    horizontal_scores = jnp.zeros((masks[0].shape[0], 512))
    horizontal_masks = jnp.zeros((masks[0].shape[0], 1))

    for e, n, m, v, h in zip(image_embeddings, embedding_norms, masks, vertical, horizontal):
#         sim = (jnp.matmul(e, text_embedding.T)) #  + 1) / 2
        
#         sim = jax.nn.relu(sim)
        
        # if full_sim is None:
        #     full_sim = sim
        # sim = jax.nn.relu(sim - full_sim)
        emb = jnp.expand_dims(e, axis=0) #* n
        
        if v:
            vm = jnp.any(m, axis=0)
            vertical_scores = vertical_scores + (emb * vm) / jnp.mean(vm)
            vertical_masks = vertical_masks + vm / jnp.mean(vm)
        if h:
            hm = jnp.any(m, axis=1)
            horizontal_scores = horizontal_scores + (emb * hm) / jnp.mean(hm)
            horizontal_masks = horizontal_masks + hm / jnp.mean(hm)
                
    
    embs_1 = jnp.expand_dims((vertical_scores), axis=0) * jnp.expand_dims(jnp.abs(horizontal_scores), axis=1)
    embs_2 = jnp.expand_dims(jnp.abs(vertical_scores), axis=0) * jnp.expand_dims((horizontal_scores), axis=1)
    full_embs = jnp.minimum(embs_1, embs_2)
    mask_sum = jnp.expand_dims(vertical_masks + 1, axis=0) * jnp.expand_dims(horizontal_masks + 1, axis=1)
    full_embs = (full_embs / mask_sum)
    
    orig_shape = full_embs.shape
    sims = jnp.matmul(jnp.reshape(full_embs, (-1, 512)), text_embeddings.T)
    score = jnp.reshape(sims, (*orig_shape[:2], 1))
    
    for i in range(iterations):
        score = jnp.clip(score - jnp.mean(score), 0, jnp.inf)
        
    score = (score - jnp.min(score)) / (jnp.max(score) - jnp.min(score))
    
    print(jnp.min(score), jnp.max(score))
    
    return np.asarray(score), input_image


def app():
    st.title("Zero-Shot Localization")
    st.markdown(
        """

        ### 👋 Ciao!

        Here you can find an example for zero-shot localization that will show you where in an image the model sees an object.
        
        The object location is computed by masking different areas of the image and looking at 
        how the similarity to the image description changes. If you want to have a look at the implementation in detail,
        you can find it in [this Colab](https://colab.research.google.com/drive/10neENr1DEAFq_GzsLqBDo0gZ50hOhkOr?usp=sharing).
        
        On the two parameters: 
        + the *pixel size* defines the resolution of the localization map. A pixel size of 15 means 
        that 15 pixels in the original image will form 1 pixel in the heatmap. 
        + The *refinement iterations* are just a cheap operation to reduce background noise. Too few iterations will leave a lot of noise. 
        Too many will shrink the heatmap too much.


        🤌 Italian mode on! 🤌

        For example, try typing "gatto" (cat) or "cane" (dog) in the space for label and click "locate"!

        """
    )

    image_url = st.text_input(
        "You can input the URL of an image here...",
        value="https://www.tuttosuigatti.it/files/styles/full_width/public/images/featured/205/cani-e-gatti.jpg",
    )

    MAX_ITER = 1

    col1, col2 = st.columns([0.75, 0.25])

    with col2:
        pixel_size = st.selectbox("Pixel Size", options=range(5, 26, 5), index=3)

        iterations = st.selectbox("Refinement Steps", options=range(0, 6, 1), index=0)

        compute = st.button("LOCATE")

    with col1:
        caption = st.text_input(f"Insert label...")

    if compute:

        with st.spinner("Waiting for resources..."):
            sleep_time = 5
            while psutil.cpu_percent() > 50:
                time.sleep(sleep_time)

        if not caption or not image_url:
            st.error("Please choose one image and at least one label")
        else:
            with st.spinner(
                "Computing... This might take up to a few minutes depending on the current load 😕  \n"
                "Otherwise, you can use this [Colab notebook](https://colab.research.google.com/drive/10neENr1DEAFq_GzsLqBDo0gZ50hOhkOr?usp=sharing)"
            ):
                heatmap, image = get_heatmap(image_url, caption, pixel_size, iterations)

                with col1:
                    st.image(image, use_column_width=True)
                    st.image(heatmap, use_column_width=True)
                    st.image(np.asarray(image) / 255.0 * heatmap, use_column_width=True)
        gc.collect()

    elif image_url:
        image = requests.get(
            image_url, 
            headers=headers,
            stream=True,
        ).raw
        image = Image.open(image).convert("RGB")
        with col1:
            st.image(image)
