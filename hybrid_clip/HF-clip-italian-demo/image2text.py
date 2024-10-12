import streamlit as st
from text2image import get_model, get_tokenizer, get_image_transform
from utils import text_encoder, image_encoder
from PIL import Image
from jax import numpy as jnp
from io import BytesIO
import pandas as pd
import requests
import jax
import gc

headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}


def app():
    st.title("From Image to Text")
    st.markdown(
        """

        ### 👋 Ciao!

        Here you can find the captions or the labels that are most related to a given image. It is a zero-shot
        image classification task!

        🤌 Italian mode on! 🤌
        
        For example, try typing "gatto" (cat) in the space for label1 and "cane" (dog) in the space for label2 and click
        "classify"!

        """
    )

    image_url = st.text_input(
        "You can input the URL of an image",
        value="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Ragdoll%2C_blue_mitted.JPG/1280px-Ragdoll%2C_blue_mitted.JPG",
    )

    MAX_CAP = 4

    col1, col2 = st.columns([0.75, 0.25])

    with col2:
        captions_count = st.selectbox(
            "Number of labels", options=range(1, MAX_CAP + 1), index=1
        )
        compute = st.button("CLASSIFY")

    with col1:
        captions = list()
        for idx in range(min(MAX_CAP, captions_count)):
            captions.append(st.text_input(f"Insert label {idx+1}"))

    if compute:
        captions = [c for c in captions if c != ""]

        if not captions or not image_url:
            st.error("Please choose one image and at least one label")
        else:
            with st.spinner("Computing..."):
                model = get_model()
                tokenizer = get_tokenizer()

                text_embeds = list()
                for i, c in enumerate(captions):
                    text_embeds.extend(text_encoder(c, model, tokenizer)[0])

                text_embeds = jnp.array(text_embeds)
                response = requests.get(image_url, headers=headers, stream=True)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                transform = get_image_transform(model.config.vision_config.image_size)
                image_embed, _ = image_encoder(transform(image), model)

                # we could have a softmax here
                cos_similarities = jax.nn.softmax(
                    jnp.matmul(image_embed, text_embeds.T)
                )

                chart_data = pd.Series(cos_similarities[0], index=captions)

                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(chart_data)

                with col2:
                    st.image(image, use_column_width=True)
        gc.collect()

    elif image_url:
        response = requests.get(image_url, headers=headers, stream=True)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(image)
