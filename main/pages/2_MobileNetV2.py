import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import requests

st.set_page_config(page_title="MobileNetV2 Transfer Learning", layout="wide")

st.title("Transfer Learning: MobileNetV2")
st.write("Memahami bagaimana Transfer Learning menggunakan model pre-trained untuk task baru")

def download_model(url, filename):
    if os.path.exists(filename):
        return True
    try:
        with st.spinner(f"Downloading model {filename}..."):
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
            
            if os.path.getsize(filename) < 1000000:
                st.error("Downloaded file too small, possibly corrupted")
                os.remove(filename)
                return False
            return True
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

@st.cache_resource
def load_model():
    model_url = "https://github.com/Arfazrll/CA-Modul03-HandsOn/releases/download/modelresultNetv2/best_transfer_model.keras"
    model_path = 'best_transfer_model.keras'
    
    if not os.path.exists(model_path):
        st.info("Model not found locally. Downloading from GitHub...")
        if not download_model(model_url, model_path):
            return create_untrained_model(), False
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        st.success("MobileNetV2 model loaded successfully")
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                st.info("Corrupted file removed. Please refresh to retry download.")
            except:
                pass
        return create_untrained_model(), False

def create_untrained_model():
    st.warning("Using untrained transfer learning model. Predictions will not be accurate.")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def preprocess_mobilenetv2(image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array.copy())
    return img_resized, img_array, img_preprocessed

model, is_trained = load_model()
class_names = ['Cheetah', 'Hyena']

st.markdown("---")

st.header("What is Transfer Learning?")
st.write("Transfer Learning is a machine learning technique where we use a model already trained on a large dataset like ImageNet with 14 million images and adapt it for a new task with a smaller dataset. Benefits: No need to train from scratch, faster and more efficient, works well with small datasets, leverages previously learned knowledge.")

st.markdown("---")

st.header("MobileNetV2 Pre-trained on ImageNet")
st.write("MobileNetV2 is a model already trained to recognize 1000 object categories from the ImageNet dataset. This model has learned to recognize: Basic features like lines, edges, colors and textures, Intermediate features like shapes and patterns, Complex features like object parts and shape combinations. We will use this knowledge to distinguish between Cheetahs and Hyenas.")

col1, col2 = st.columns(2)

with col1:
    st.info("Frozen Layers (Not Trained): Base MobileNetV2, Already capable of extracting general features, Parameters: 2.2 million")

with col2:
    if is_trained:
        st.success("Trained Layers: Global Average Pooling, Dense Layer 128 neurons, Output Layer 2 classes, Model IS TRAINED")
    else:
        st.warning("Layers to Train: Global Average Pooling, Dense Layer 128 neurons, Output Layer 2 classes, Model NOT YET TRAINED")

st.markdown("---")

if is_trained:
    st.success("Model ready for accurate predictions")
else:
    st.warning("Model not fine-tuned for Hyena vs Cheetah. Application still functions for learning but predictions may not be accurate.")

st.markdown("---")

st.header("Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose Cheetah or Hyena image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("After Preprocessing")
        img_resized, img_array, img_preprocessed = preprocess_mobilenetv2(image)
        st.image(img_resized, use_column_width=True)
        st.caption("Size: 224x224 pixels")
        st.caption(f"Original range: [0, 255]")
        st.caption(f"Preprocessed range: [{img_preprocessed.min():.2f}, {img_preprocessed.max():.2f}]")
        st.info("MobileNetV2 uses special preprocessing not just division by 255")
    
    st.markdown("---")
    
    st.header("RGB Channel Analysis")
    st.write("MobileNetV2 accepts 3-channel RGB input. See the color intensity distribution in your image.")
    
    img_display = np.array(img_resized)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Red Channel")
        fig_r = go.Figure(data=go.Heatmap(
            z=img_display[:,:,0][::-1],
            colorscale='Reds',
            showscale=True
        ))
        fig_r.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_r, use_column_width=True)
        st.metric("Mean Red", f"{np.mean(img_display[:,:,0]):.2f}")
    
    with col2:
        st.subheader("Green Channel")
        fig_g = go.Figure(data=go.Heatmap(
            z=img_display[:,:,1][::-1],
            colorscale='Greens',
            showscale=True
        ))
        fig_g.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_g, use_column_width=True)
        st.metric("Mean Green", f"{np.mean(img_display[:,:,1]):.2f}")
    
    with col3:
        st.subheader("Blue Channel")
        fig_b = go.Figure(data=go.Heatmap(
            z=img_display[:,:,2][::-1],
            colorscale='Blues',
            showscale=True
        ))
        fig_b.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_b, use_column_width=True)
        st.metric("Mean Blue", f"{np.mean(img_display[:,:,2]):.2f}")
    
    st.markdown("---")
    
    st.header("Pixel Intensity Distribution")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=img_display[:,:,0].flatten(),
        name='Red',
        marker_color='red',
        opacity=0.7,
        nbinsx=50
    ))
    fig_hist.add_trace(go.Histogram(
        x=img_display[:,:,1].flatten(),
        name='Green',
        marker_color='green',
        opacity=0.7,
        nbinsx=50
    ))
    fig_hist.add_trace(go.Histogram(
        x=img_display[:,:,2].flatten(),
        name='Blue',
        marker_color='blue',
        opacity=0.7,
        nbinsx=50
    ))
    
    fig_hist.update_layout(
        barmode='overlay',
        xaxis_title="Pixel Intensity (0-255)",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig_hist, use_column_width=True)
    st.caption("Histogram shows intensity distribution for each color channel")
    
    st.markdown("---")
    
    st.header("Prediction and Classification")
    
    button_label = "Run Prediction with MobileNetV2" if is_trained else "Try Prediction (Model Not Fine-tuned)"
    
    if st.button(button_label, type="primary"):
        with st.spinner("Processing image through MobileNetV2..."):
            img_input = np.expand_dims(img_preprocessed, axis=0)
            predictions = model.predict(img_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
        
        if is_trained:
            st.success(f"Prediction: **{predicted_class}** with confidence **{confidence:.2f}%**")
        else:
            st.warning(f"Prediction (MAY NOT BE ACCURATE): **{predicted_class}** - Confidence: {confidence:.2f}%")
            st.error("Model not fine-tuned for Hyena vs Cheetah. Prediction may not be reliable")
        
        st.subheader("Probability Distribution")
        st.write("Model gives probabilities for each class. Output layer uses softmax activation which converts values to probabilities totaling 100%." if is_trained else "These probabilities are NOT RELIABLE because model has not been trained to distinguish Hyena and Cheetah")
        
        fig_pred = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=predictions[0] * 100,
                marker_color=['#ff9800', '#2196f3'],
                text=[f'{val:.2f}%' for val in predictions[0] * 100],
                textposition='auto'
            )
        ])
        fig_pred.update_layout(
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_pred, use_column_width=True)
        
        st.markdown("---")
        
        st.header("Feature Extraction from MobileNetV2")
        st.write("MobileNetV2 extracts 1280 features from input images. These features represent important characteristics learned from ImageNet. See the most important features.")
        
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-3].output
        )
        features = feature_extractor.predict(img_input, verbose=0)
        
        st.write(f"**Extracted feature shape:** {features.shape}")
        st.caption("(1, 1280) means 1280 features for 1 image")
        
        feature_values = features[0]
        top_indices = np.argsort(np.abs(feature_values))[-30:][::-1]
        
        fig_feat = go.Figure(data=[
            go.Bar(
                x=list(range(len(top_indices))),
                y=np.abs(feature_values[top_indices]),
                marker_color='#9c27b0',
                text=[f'{val:.3f}' for val in feature_values[top_indices]],
                textposition='outside'
            )
        ])
        fig_feat.update_layout(
            title="Top 30 Most Important Features (By Magnitude)",
            xaxis_title="Feature Index",
            yaxis_title="Absolute Value",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_feat, use_column_width=True)
        
        if is_trained:
            st.write("Features with large magnitude positive or negative have stronger influence in determining final classification between Cheetah and Hyena.")
        else:
            st.warning("These features are from ImageNet pre-training. After fine-tuning these features will be adjusted to distinguish Cheetah and Hyena.")
        
        st.markdown("---")
        
        st.header("Transfer Learning Process")
        st.write("""
        Complete flow of how your image is processed:
        
        Step 1: Input Processing - Image resized to 224x224 pixels, Special MobileNetV2 preprocessing not just division by 255, Normalization with mean subtraction and scaling
        
        Step 2: Feature Extraction MobileNetV2 - Image goes through 53 convolutional layers, Produces feature maps 7x7x1280, Extracts features from simple to complex, Uses depthwise separable convolutions
        
        Step 3: Global Average Pooling - Feature maps 7x7x1280 averaged, Produces vector of 1280 features, Reduces overfitting
        
        Step 4: Dense Classification - 1280 features to 128 neurons with ReLU, Dropout 50% for regularization, 128 neurons to 2 outputs Cheetah and Hyena
        
        Step 5: Softmax - Converts output to probabilities, Prediction equals class with highest probability
        """)
        
        st.markdown("---")
        
        st.header("Architecture Visualization")
        
        layers_info = [
            {"name": "Input Image", "size": "224x224x3", "params": "0", "desc": "RGB with MobileNetV2 preprocessing"},
            {"name": "MobileNetV2 Base", "size": "7x7x1280", "params": "2.2M", "desc": "Feature Extraction frozen"},
            {"name": "Global Avg Pooling", "size": "1280", "params": "0", "desc": "Spatial dimension reduction"},
            {"name": "Dense + ReLU", "size": "128", "params": "163K", "desc": "Feature combination"},
            {"name": "Dropout 0.5", "size": "128", "params": "0", "desc": "Regularization"},
            {"name": "Output + Softmax", "size": "2", "params": "258", "desc": "Classification Cheetah Hyena"}
        ]
        
        for i, layer in enumerate(layers_info):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
            with col1:
                st.write(f"**{i+1}. {layer['name']}**")
            with col2:
                st.write(f"Size: {layer['size']}")
            with col3:
                st.write(f"Params: {layer['params']}")
            with col4:
                st.write(f"{layer['desc']}")
            
            if i < len(layers_info) - 1:
                st.write("↓")
        
        st.markdown("---")
        
        st.header("Why is MobileNetV2 Effective?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Depthwise Separable Convolution")
            st.write("MobileNetV2 uses special technique that separates: Depthwise Convolution processes each channel separately, Pointwise Convolution combines channels with 1x1 filter. Result: 8-9x fewer parameters than regular CNN")
        
        with col2:
            st.subheader("Inverted Residual Blocks")
            st.write("Unique structure that: Expands features with 1x1 conv, Depthwise 3x3 conv for spatial features, Linear projection for output. Result: Efficient for mobile and embedded devices")
        
        st.markdown("---")
        

else:
    st.info("Please upload Cheetah or Hyena image to start prediction and see visualizations")
    
    st.markdown("---")
    
    st.header("Comparison: CNN vs Transfer Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CNN from Scratch")
        st.write("Advantages: Full control over architecture, Optimized for specific dataset. Disadvantages: Needs large dataset over 10000 images, Long training hours to days, Requires large computational resources, Prone to overfitting on small datasets")
    
    with col2:
        st.subheader("Transfer Learning")
        st.write("Advantages: Works with small datasets less than 1000 images, Fast training minutes to hours, Less computational resources, High accuracy with minimal data. Disadvantages: Limited to similar domains, Larger model size")
    
    st.markdown("---")
    
    st.header("Dataset: Cheetah vs Hyena")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cheetah")
        st.write("Visual characteristics: Small round black spots, Black line from eyes to mouth tear mark, Slimmer and more athletic body, Bright golden brown color")
    
    with col2:
        st.subheader("Hyena")
        st.write("Visual characteristics: Irregular spots or stripes, No tear mark, Larger and more robust body, Darker color grayish brown")
    
    st.write("Trained MobileNetV2 model can recognize these subtle differences because it already has knowledge about textures patterns and shapes from ImageNet training.")