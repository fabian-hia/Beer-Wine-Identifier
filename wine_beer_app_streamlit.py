import os
import streamlit as st
from PIL import Image
from fastai.vision.all import *
from fastai.vision.widgets import *
import matplotlib.pyplot as plt

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Function to load the model
@st.cache_resource()
def load_model():
    model_path = 'model_beerwine.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    return load_learner(model_path)

# Function to classify an image
def classify_image(image, model, categories):
    try:
        image = image.convert('RGB').resize((192, 192))
        pred, idx, probs = model.predict(image)
        return dict(zip(categories, map(float, probs)))
    except Exception as e:
        st.error(f"An error occurred during image classification: {e}")
        return None

# Function to plot predictions
def plot_predictions(predictions):
    fig, ax = plt.subplots()
    ax.bar(predictions.keys(), predictions.values(), color=['#FF9999','#66B3FF'])
    ax.set_xlabel('Category')
    ax.set_ylabel('Probability')
    ax.set_title('Predictions')
    st.pyplot(fig)

# Function to handle example images
def show_example_images(model, categories):
    st.write("## Example Images")
    examples = [os.path.join('.', example_name) 
                for example_name in ['Beer.jpg', 'Wine.jpg', 'Unknown.jpg']]
    
    selected_example = st.radio("Click to see predictions:", examples, format_func=lambda x: os.path.basename(x))
    
    if selected_example:
        example_image = Image.open(selected_example)
        st.image(example_image, caption=os.path.basename(selected_example), use_column_width=True)
        example_predictions = classify_image(example_image, model, categories)
        if example_predictions:
            st.write(example_predictions)
            st.session_state.predictions = example_predictions  # Save predictions to session state

# Main app function
def main():
    st.title("Image Classification: Beer or Wine")
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return
    
    categories = ('beer', 'wine')

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("Classifying...")
                predictions = classify_image(image, model, categories)
                if predictions:
                    st.write(predictions)
                    st.session_state.predictions = predictions  # Save predictions to session state
            except Exception as e:
                st.error(f"An error occurred: {e}")

        show_example_images(model, categories)

    with col2:
        if 'predictions' in st.session_state:
            plot_predictions(st.session_state.predictions)

if __name__ == "__main__":
    main()
