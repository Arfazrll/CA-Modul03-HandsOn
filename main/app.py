import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="Deep Learning Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .card {
            padding: 2rem;
            border-radius: 10px;
            background: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .feature-box {
            padding: 1.5rem;
            border-left: 4px solid #667eea;
            background: white;
            margin: 1rem 0;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Deep Learning Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualisasi Real-Time Arsitektur Neural Network</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
        <div class="card">
            <h2 style="text-align: center; color: #667eea;">Selamat Datang</h2>
            <p style="text-align: center; font-size: 1.1rem; color: #555;">
                Platform ini memungkinkan Anda untuk memahami dan mengeksplorasi 
                cara kerja Deep Learning dalam klasifikasi gambar secara visual dan interaktif.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-box">
            <h3 style="color: #667eea;">🔷 Custom CNN Architecture</h3>
            <p><strong>Dataset:</strong> Rock Paper Scissors</p>
            <p><strong>Classes:</strong> Rock, Paper, Scissors</p>
            <p><strong>Image Size:</strong> 150x150 pixels</p>
            <h4>Fitur:</h4>
            <ul>
                <li>Visualisasi RGB Channels</li>
                <li>Konvolusi Real-Time</li>
                <li>Feature Maps Analysis</li>
                <li>Arsitektur Neural Network</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-box">
            <h3 style="color: #764ba2;">🔶 Transfer Learning MobileNetV2</h3>
            <p><strong>Dataset:</strong> Hyena vs Cheetah</p>
            <p><strong>Classes:</strong> Hyena, Cheetah</p>
            <p><strong>Image Size:</strong> 224x224 pixels</p>
            <h4>Fitur:</h4>
            <ul>
                <li>Pre-trained Weights</li>
                <li>Feature Extraction</li>
                <li>Confidence Visualization</li>
                <li>Transfer Learning Pipeline</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
    <div class="card">
        <h2 style="text-align: center; color: #667eea;">Cara Kerja Deep Learning</h2>
        <br>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">1. Input Layer</h3>
            <p>Gambar diproses menjadi matrix RGB dengan dimensi (Height x Width x 3)</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">2. Convolution</h3>
            <p>Filter kernel mengekstrak fitur seperti edges, textures, dan patterns</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #667eea;">3. Classification</h3>
            <p>Fully Connected Layer menghasilkan probabilitas untuk setiap kelas</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3 style="color: #667eea;">Pilih Model dari Sidebar</h3>
        <p style="font-size: 1.1rem; color: #666;">
            Gunakan menu sidebar di sebelah kiri untuk memilih antara 
            <strong>CNN Classification</strong> atau <strong>MobileNetV2 Transfer Learning</strong>
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h4>Tentang Aplikasi</h4>
        <p>Platform visualisasi Deep Learning untuk memahami cara kerja Neural Network secara real-time</p>
    </div>
""", unsafe_allow_html=True)