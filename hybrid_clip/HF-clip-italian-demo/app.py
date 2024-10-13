import streamlit as st
import image2text
import text2image
import localization
import home
import examples
from PIL import Image

PAGES = {
    "Introduction": home,
    "Text to Image": text2image,
    "Image to Text": image2text,
    "Localization": localization,
    "Gallery": examples,
}

st.sidebar.title("Explore our CLIP-Italian demo")

logo = Image.open("static/img/clip_italian_logo.png")
st.sidebar.image(logo, caption="CLIP-Italian logo")

page = st.sidebar.radio("sb_radio_label", list(PAGES.keys()), label_visibility='hidden')
PAGES[page].app()
