import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import cv2
from scipy.ndimage import convolve
import requests
import zipfile
import os

st.set_page_config(page_title="CNN Classification", page_icon="🔷", layout="wide")

st.markdown("""
    <style>
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        .metric-card {
            padding: 1.5rem;
            border-radius: 8px;
            background: #f8f9fa;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_and_extract_dataset():
    url = "https://github.com/Arfazrll/CA-Modul03-HandsOn/releases/download/ConvolutionalNeuralNetwork/rockpaperscissors.zip"
    zip_path = "rockpaperscissors.zip"
    extract_path = "data"
    
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path):
            with st.spinner("Downloading dataset..."):
                response = requests.get(url, stream=True)
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        with st.spinner("Extracting dataset..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    
    return extract_path

@st.cache_resource
def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def visualize_rgb_channels(image):
    img_array = np.array(image)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(z=img_array[:,:,0], colorscale='Reds', name='Red', showscale=False))
    
    return fig

def visualize_convolution(image, kernel_type='edge_detection'):
    img_array = np.array(image.convert('L'))
    
    kernels = {
        'edge_detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'blur': np.ones((3, 3)) / 9,
        'vertical_edge': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'horizontal_edge': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    }
    
    kernel = kernels[kernel_type]
    convolved = convolve(img_array, kernel)
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    
    return convolved, kernel

def create_architecture_flow():
    layers = [
        {"name": "Input\n150x150x3", "neurons": 150, "color": "#667eea"},
        {"name": "Conv2D\n32 filters", "neurons": 148, "color": "#764ba2"},
        {"name": "MaxPool\n74x74x32", "neurons": 74, "color": "#f093fb"},
        {"name": "Conv2D\n64 filters", "neurons": 72, "color": "#4facfe"},
        {"name": "MaxPool\n36x36x64", "neurons": 36, "color": "#00f2fe"},
        {"name": "Conv2D\n128 filters", "neurons": 34, "color": "#43e97b"},
        {"name": "MaxPool\n17x17x128", "neurons": 17, "color": "#38f9d7"},
        {"name": "Flatten\n36992", "neurons": 100, "color": "#fa709a"},
        {"name": "Dense\n512", "neurons": 80, "color": "#fee140"},
        {"name": "Output\n3 classes", "neurons": 3, "color": "#30cfd0"}
    ]
    
    fig = go.Figure()
    
    for i, layer in enumerate(layers):
        x_pos = i * 1.5
        fig.add_trace(go.Scatter(
            x=[x_pos] * layer["neurons"],
            y=np.linspace(0, 10, layer["neurons"]),
            mode='markers',
            marker=dict(size=8, color=layer["color"], opacity=0.6),
            name=layer["name"],
            showlegend=True
        ))
        
        if i < len(layers) - 1:
            for start_y in np.linspace(0, 10, min(5, layer["neurons"])):
                for end_y in np.linspace(0, 10, min(5, layers[i+1]["neurons"])):
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos + 1.5],
                        y=[start_y, end_y],
                        mode='lines',
                        line=dict(color='lightgray', width=0.5),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    fig.update_layout(
        title="CNN Architecture Visualization",
        showlegend=True,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

st.markdown('<h1 style="text-align: center; color: #667eea;">🔷 CNN Classification</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Rock Paper Scissors Detection</p>', unsafe_allow_html=True)

st.markdown("---")

try:
    data_path = download_and_extract_dataset()
except Exception as e:
    st.error(f"Error downloading dataset: {str(e)}")
    data_path = None

model = build_cnn_model()

class_names = ['paper', 'rock', 'scissors']

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        st.markdown("### 🎨 RGB Channel Visualization")
        img_array = np.array(image.resize((150, 150)))
        
        fig_rgb = go.Figure()
        
        fig_rgb.add_trace(go.Heatmap(
            z=img_array[:,:,0][::-1],
            colorscale='Reds',
            name='Red Channel',
            showscale=False
        ))
        
        fig_rgb.update_layout(
            title="Red Channel",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_rgb, use_container_width=True)
        
        fig_rgb_g = go.Figure()
        fig_rgb_g.add_trace(go.Heatmap(
            z=img_array[:,:,1][::-1],
            colorscale='Greens',
            showscale=False
        ))
        fig_rgb_g.update_layout(
            title="Green Channel",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_rgb_g, use_container_width=True)
        
        fig_rgb_b = go.Figure()
        fig_rgb_b.add_trace(go.Heatmap(
            z=img_array[:,:,2][::-1],
            colorscale='Blues',
            showscale=False
        ))
        fig_rgb_b.update_layout(
            title="Blue Channel",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_rgb_b, use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🔍 Convolution Operations")
        
        kernel_type = st.selectbox(
            "Select Kernel Type",
            ['edge_detection', 'sharpen', 'blur', 'vertical_edge', 'horizontal_edge']
        )
        
        convolved_img, kernel = visualize_convolution(image, kernel_type)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Kernel Matrix**")
            kernel_df = kernel
            fig_kernel = go.Figure(data=[go.Heatmap(
                z=kernel,
                colorscale='RdBu',
                text=kernel,
                texttemplate='%{text:.1f}',
                textfont={"size": 16}
            )])
            fig_kernel.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_kernel, use_container_width=True)
        
        with col_b:
            st.markdown("**Convolution Result**")
            st.image(convolved_img, use_container_width=True)
        
        st.markdown("### 🎯 Prediction")
        
        if st.button("Predict Image", type="primary", use_container_width=True):
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner("Processing..."):
                predictions = model.predict(img_array, verbose=0)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction: {predicted_class.upper()}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Confidence Distribution")
            
            fig_pred = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=predictions[0] * 100,
                    marker_color=['#667eea', '#764ba2', '#f093fb'],
                    text=[f'{val:.2f}%' for val in predictions[0] * 100],
                    textposition='auto',
                )
            ])
            
            fig_pred.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("### 🧠 Feature Extraction Process")
            
            layer_outputs = []
            layer_names = []
            
            for layer in model.layers[:4]:
                if 'conv' in layer.name or 'pool' in layer.name:
                    intermediate_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=layer.output
                    )
                    layer_output = intermediate_model.predict(img_array, verbose=0)
                    layer_outputs.append(layer_output)
                    layer_names.append(layer.name)
            
            cols = st.columns(len(layer_outputs))
            
            for idx, (col, output, name) in enumerate(zip(cols, layer_outputs, layer_names)):
                with col:
                    st.markdown(f"**{name}**")
                    feature_map = output[0, :, :, 0]
                    
                    fig_fm = go.Figure(data=[go.Heatmap(
                        z=feature_map,
                        colorscale='Viridis',
                        showscale=False
                    )])
                    fig_fm.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_fm, use_container_width=True)

st.markdown("---")

st.markdown("### 🏗️ Network Architecture")
arch_fig = create_architecture_flow()
st.plotly_chart(arch_fig, use_container_width=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea;">32-128</h3>
            <p>Convolutional Filters</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #764ba2;">150x150</h3>
            <p>Input Image Size</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f093fb;">3 Classes</h3>
            <p>Rock, Paper, Scissors</p>
        </div>
    """, unsafe_allow_html=True)