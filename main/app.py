import streamlit as st

st.set_page_config(
    page_title="Deep Learning Image Classifier",
    layout="wide"
)

st.title("Deep Learning Image Classifier")
st.write("A platform to understand how CNN and Transfer Learning work for image classification.")

st.markdown("---")

st.header("Deep Learning Visualization")

st.write("""
This application is designed to help you understand Deep Learning concepts through interactive visualizations.
You can see how neural network models process images and make predictions.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CNN Classification")
    st.write("**Dataset:** Rock Paper Scissors")
    st.write("**Classes:** Rock, Paper, Scissors")
    st.write("**Input Size:** 150x150 pixels")
    st.write("""
    On this page, you will learn:
    - How a CNN processes images
    - How convolution and pooling work
    - How feature extraction happens
    - How predictions are made based on feature maps
    """)

with col2:
    st.subheader("MobileNetV2 Transfer Learning")
    st.write("**Dataset:** Hyena vs Cheetah")
    st.write("**Classes:** Hyena, Cheetah")
    st.write("**Input Size:** 224x224 pixels")
    st.write("""
    On this page, you will learn:
    - The concept of Transfer Learning
    - How a pre-trained ImageNet model works
    - The feature extraction process
    - How fine-tuning adapts the model to a new task
    """)

st.markdown("---")

st.info("Select one of the models from the sidebar to get started.")