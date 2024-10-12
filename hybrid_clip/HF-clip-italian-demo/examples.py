from home import read_markdown_file
import streamlit as st


def app():
    st.title("Gallery")
    st.write(
        """

        Even though we trained the Italian CLIP model on way less examples than the original
        OpenAI's CLIP, our training choices and quality datasets led to impressive results.
        Here, we present some of them.
        
        """
    )

    st.markdown("### 1. Actors in Scenes")
    st.markdown("These examples were taken from the CC dataset.")

    st.subheader("Una coppia")
    st.markdown("*A couple*")
    st.image("static/img/examples/couple_0.jpeg", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("Una coppia con il tramonto sullo sfondo")
    col1.markdown("*A couple with the sunset in the background*")
    col1.image("static/img/examples/couple_1.jpeg", use_column_width=True)

    col2.subheader("Una coppia che passeggia sulla spiaggia")
    col2.markdown("*A couple walking on the beach*")
    col2.image("static/img/examples/couple_2.jpeg", use_column_width=True)

    st.subheader("Una coppia che passeggia sulla spiaggia al tramonto")
    st.markdown("*A couple walking on the beach at sunset*")
    st.image("static/img/examples/couple_3.jpeg", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("Un bambino con un biberon")
    col1.markdown("*A baby with a bottle*")
    col1.image("static/img/examples/bambino_biberon.jpeg", use_column_width=True)

    col2.subheader("Un bambino con un gelato in spiaggia")
    col2.markdown("*A child with an ice cream on the beach*")
    col2.image(
        "static/img/examples/bambino_gelato_spiaggia.jpeg", use_column_width=True
    )

    st.markdown("### 2. Dresses")
    st.markdown("These examples were taken from the Unsplash dataset.")

    col1, col2 = st.columns(2)
    col1.subheader("Un vestito primaverile")
    col1.markdown("*A dress for the spring*")
    col1.image("static/img/examples/vestito1.png", use_column_width=True)

    col2.subheader("Un vestito autunnale")
    col2.markdown("*A dress for the autumn*")
    col2.image("static/img/examples/vestito_autunnale.png", use_column_width=True)

    st.markdown("### 3. Chairs with different styles")
    st.markdown("These examples were taken from the CC dataset.")

    col1, col2 = st.columns(2)
    col1.subheader("Una sedia semplice")
    col1.markdown("*A simple chair*")
    col1.image("static/img/examples/sedia_semplice.jpeg", use_column_width=True)

    col2.subheader("Una sedia regale")
    col2.markdown("*A royal chair*")
    col2.image("static/img/examples/sedia_regale.jpeg", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("Una sedia moderna")
    col1.markdown("*A modern chair*")
    col1.image("static/img/examples/sedia_moderna.jpeg", use_column_width=True)

    col2.subheader("Una sedia rustica")
    col2.markdown("*A rustic chair*")
    col2.image("static/img/examples/sedia_rustica.jpeg", use_column_width=True)

    st.markdown("## Localization")

    st.subheader("Un gatto")
    st.markdown("*A cat*")
    st.image("static/img/examples/un_gatto.png", use_column_width=True)

    st.subheader("Un gatto")
    st.markdown("*A cat*")
    st.image("static/img/examples/due_gatti.png", use_column_width=True)

    st.subheader("Un bambino")
    st.markdown("*A child*")
    st.image("static/img/examples/child_on_slide.png", use_column_width=True)

    st.subheader("A complex example: Uno squalo / un cavallo")
    st.markdown("*A shark / a horse*")
    st.image("static/img/examples/cavallo_squalo.png", use_column_width=True)

    st.markdown("## Image Classification")
    st.markdown(
        "We report this cool example provided by the "
        "[DALLE-mini team](https://github.com/borisdayma/dalle-mini). "
        "Is the DALLE-mini logo an *avocado* or an armchair (*poltrona*)?"
    )

    st.image("static/img/examples/dalle_mini.png", use_column_width=True)
    st.markdown(
        "It seems it's half an armchair and half an avocado! We thank the DALL-E mini team for the great idea :)"
    )

    st.subheader("A more classic example")
    st.markdown("Is this a pizza, a dish of pasta or a cat?")
    st.image("static/img/examples/pizza.png", use_column_width=True)
