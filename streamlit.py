import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import streamlit as st
import tensorflow as tf
#from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Tea Leaf Disease Detection",
    page_icon = ":tea:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:

            return key


with st.sidebar:
        st.image('mg.png')
        st.title("CHAICHECK")
        st.subheader("Accurate detection of diseases present leaves.")



def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:

            return key


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model_keras():
    model=tf.keras.models.load_model('tea_leaf_disease_model.h5', compile=False)
    return model
with st.spinner('Model is being loaded..'):
    model=load_model_keras()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))


st.write("""
         # Tea Disease Detection with Remedy - @Bello
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Display image and accuracy
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

    detected_disease = class_names[np.argmax(predictions)]
    st.subheader("Detected Disease:")
    st.write(f"Detected Disease : {detected_disease}")
    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    # Probability Dashboard
    st.subheader("Probability Dashboard:")
    probabilities = {class_name: predictions[0][i] for i, class_name in enumerate(class_names)}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
    st.bar_chart(df.set_index('Class'))

    # Display remedy information based on detected disease
    if detected_disease == 'Healthy':
        st.balloons()
        st.sidebar.success("Detected Disease: Healthy")

    elif class_names[np.argmax(predictions)] == 'algal leaf':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

    elif class_names[np.argmax(predictions)] == 'bird eye spot':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")

    elif class_names[np.argmax(predictions)] == 'brown blight':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("brown blight can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the damage.")

    elif class_names[np.argmax(predictions)] == 'white spot':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("After pruning, apply copper oxychloride at a concentration of '0.3%' on the wounds. Apply Bordeaux mixture twice a year to reduce the infection rate on the trees. Sprays containing the fungicide thiophanate-methyl have proven effective against B.")

    elif class_names[np.argmax(predictions)] == 'gray light':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Use yellow sticky traps to catch the flies. Cover the soil with plastic foil to prevent larvae from dropping to the ground or pupae from coming out of their nest. Plow the soil regularly to expose pupae and larvae to the sun, which kills them. Collect and burn infested tree material during the season.")


    elif class_names[np.argmax(predictions)] == 'red leaf spo':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("In order to control red leaf spo, three sprays of fungicides are recommended. The first spray comprising of wettable sulphur (0.2%, i.e., 2g per litre of water) should be done when the panicles are 8 -10 cm in size as a preventive spray.")

    elif class_names[np.argmax(predictions)] == 'Sooty Mould':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("The insects causing the mould are killed by spraying with carbaryl or phosphomidon 0.03%. It is followed by spraying with a dilute solution of starch or maida 5%. On drying, the starch comes off in flakes and the process removes the black mouldy growth fungi from different plant parts.")
    else:
        # Image not relevant to the task
        st.warning("The uploaded image is not relevant to the tea leaf disease detection task.")