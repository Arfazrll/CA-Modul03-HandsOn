import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import cv2
import requests
import zipfile
import os

st.set_page_config(page_title="MobileNetV2", page_icon="🔶", layout="wide")

st.markdown("""
    <style>
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
        .transfer-box {
            padding: 1.5rem;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_and_extract_dataset():
    url = "https://github.com/Arfazrll/CA-Modul03-HandsOn/releases/download/TransferlearningNetv2/hyenacheetahclass.zip"
    zip_path = "hyenacheetahclass.zip"
    extract_path = "dataset"
    
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
def build_mobilenetv2_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    return model, base_model

def visualize_feature_maps(model, image, layer_name):
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    layer_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    features = layer_model.predict(img_array, verbose=0)
    return features

def create_transfer_learning_diagram():
    fig = go.Figure()
    
    stages = [
        {"name": "ImageNet\nPre-trained", "x": 0, "y": 5, "color": "#667eea"},
        {"name": "Feature\nExtraction", "x": 2, "y": 5, "color": "#764ba2"},
        {"name": "Global\nPooling", "x": 4, "y": 5, "color": "#f093fb"},
        {"name": "Dense\nLayers", "x": 6, "y": 5, "color": "#4facfe"},
        {"name": "Output\n2 Classes", "x": 8, "y": 5, "color": "#00f2fe"}
    ]
    
    for i, stage in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=[stage["x"]],
            y=[stage["y"]],
            mode='markers+text',
            marker=dict(size=80, color=stage["color"]),
            text=stage["name"],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name=stage["name"],
            showlegend=False
        ))
        
        if i < len(stages) - 1:
            fig.add_annotation(
                x=stage["x"] + 1,
                y=stage["y"],
                ax=stage["x"] + 0.5,
                ay=stage["y"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=3,
                arrowcolor="#666"
            )
    
    fig.update_layout(
        title="Transfer Learning Pipeline",
        showlegend=False,
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 7]),
        plot_bgcolor='white'
    )
    
    return fig

def visualize_rgb_3d(image):
    img_array = np.array(image.resize((50, 50)))
    
    x, y = np.meshgrid(range(50), range(50))
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        z=img_array[:,:,0],
        x=x,
        y=y,
        colorscale='Reds',
        name='Red',
        showscale=False,
        opacity=0.7
    ))
    
    fig.update_layout(
        title="3D RGB Intensity Map (Red Channel)",
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Height",
            zaxis_title="Intensity"
        ),
        height=400
    )
    
    return fig

st.markdown('<h1 style="text-align: center; color: #f093fb;">🔶 MobileNetV2 Transfer Learning</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Hyena vs Cheetah Classification</p>', unsafe_allow_html=True)

st.markdown("---")

try:
    data_path = download_and_extract_dataset()
except Exception as e:
    st.error(f"Error downloading dataset: {str(e)}")
    data_path = None

model, base_model = build_mobilenetv2_model()

class_names = ['cheetah', 'hyena']

st.markdown("""
    <div class="transfer-box">
        <h3>Transfer Learning Concept</h3>
        <p>MobileNetV2 adalah model yang telah dilatih pada ImageNet dataset dengan 1000 kategori. 
        Kita menggunakan knowledge yang telah dipelajari untuk mengenali pola umum dalam gambar, 
        kemudian fine-tune untuk klasifikasi hyena dan cheetah.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        st.markdown("### 🎨 Image Analysis")
        
        img_array = np.array(image.resize((224, 224)))
        
        tab1, tab2, tab3 = st.tabs(["RGB Channels", "3D Visualization", "Statistics"])
        
        with tab1:
            fig_rgb_combined = go.Figure()
            
            fig_rgb_combined.add_trace(go.Heatmap(
                z=img_array[:,:,0][::-1],
                colorscale='Reds',
                name='Red',
                visible=True
            ))
            
            fig_rgb_combined.add_trace(go.Heatmap(
                z=img_array[:,:,1][::-1],
                colorscale='Greens',
                name='Green',
                visible=False
            ))
            
            fig_rgb_combined.add_trace(go.Heatmap(
                z=img_array[:,:,2][::-1],
                colorscale='Blues',
                name='Blue',
                visible=False
            ))
            
            fig_rgb_combined.update_layout(
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(label="Red", method="update", args=[{"visible": [True, False, False]}]),
                            dict(label="Green", method="update", args=[{"visible": [False, True, False]}]),
                            dict(label="Blue", method="update", args=[{"visible": [False, False, True]}])
                        ]),
                        direction="down",
                        showactive=True,
                    )
                ],
                height=400
            )
            
            st.plotly_chart(fig_rgb_combined, use_container_width=True)
        
        with tab2:
            fig_3d = visualize_rgb_3d(image)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab3:
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Mean R", f"{np.mean(img_array[:,:,0]):.2f}")
            with col_b:
                st.metric("Mean G", f"{np.mean(img_array[:,:,1]):.2f}")
            with col_c:
                st.metric("Mean B", f"{np.mean(img_array[:,:,2]):.2f}")
            
            st.markdown("**Intensity Distribution**")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=img_array[:,:,0].flatten(), name='Red', marker_color='red', opacity=0.6))
            fig_hist.add_trace(go.Histogram(x=img_array[:,:,1].flatten(), name='Green', marker_color='green', opacity=0.6))
            fig_hist.add_trace(go.Histogram(x=img_array[:,:,2].flatten(), name='Blue', marker_color='blue', opacity=0.6))
            
            fig_hist.update_layout(
                barmode='overlay',
                xaxis_title="Pixel Intensity",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🎯 Prediction")
        
        if st.button("Predict Image", type="primary", use_container_width=True):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner("Processing with MobileNetV2..."):
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
                    marker_color=['#f093fb', '#f5576c'],
                    text=[f'{val:.2f}%' for val in predictions[0] * 100],
                    textposition='auto',
                )
            ])
            
            fig_pred.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("### 🔬 Feature Extraction")
            
            st.info("MobileNetV2 mengekstrak fitur dari gambar menggunakan convolutional layers yang telah dilatih pada ImageNet")
            
            features = visualize_feature_maps(model, image, 'global_average_pooling2d')
            
            st.markdown(f"**Extracted Features Shape:** {features.shape}")
            
            feature_importance = np.abs(features[0]).flatten()
            top_indices = np.argsort(feature_importance)[-20:]
            
            fig_feat = go.Figure(data=[
                go.Bar(
                    x=top_indices,
                    y=feature_importance[top_indices],
                    marker_color='#667eea'
                )
            ])
            
            fig_feat.update_layout(
                title="Top 20 Important Features",
                xaxis_title="Feature Index",
                yaxis_title="Importance",
                height=300
            )
            
            st.plotly_chart(fig_feat, use_container_width=True)
            
            st.markdown("### 🧠 Layer Activation")
            
            try:
                layer_name = 'block_16_expand'
                layer_features = visualize_feature_maps(base_model, image, layer_name)
                
                num_filters_to_show = min(8, layer_features.shape[-1])
                cols = st.columns(4)
                
                for i in range(num_filters_to_show):
                    with cols[i % 4]:
                        feature_map = layer_features[0, :, :, i]
                        
                        fig_fm = go.Figure(data=[go.Heatmap(
                            z=feature_map,
                            colorscale='Viridis',
                            showscale=False
                        )])
                        
                        fig_fm.update_layout(
                            title=f"Filter {i+1}",
                            height=150,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig_fm, use_container_width=True)
            except:
                st.warning("Feature map visualization not available for this layer")

st.markdown("---")

st.markdown("### 🏗️ Transfer Learning Architecture")

tl_fig = create_transfer_learning_diagram()
st.plotly_chart(tl_fig, use_container_width=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f093fb;">ImageNet</h3>
            <p>Pre-trained Weights</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f5576c;">224x224</h3>
            <p>Input Image Size</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea;">2 Classes</h3>
            <p>Hyena, Cheetah</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color: #764ba2;">1280</h3>
            <p>Feature Dimensions</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

with st.expander("📚 Understanding Transfer Learning"):
    st.markdown("""
    **Transfer Learning** adalah teknik machine learning di mana model yang telah dilatih pada satu task 
    digunakan sebagai starting point untuk task yang berbeda namun related.
    
    **Keuntungan:**
    - Lebih cepat dalam training
    - Memerlukan data yang lebih sedikit
    - Performa lebih baik terutama pada dataset kecil
    - Memanfaatkan knowledge dari dataset besar (ImageNet)
    
    **MobileNetV2 Architecture:**
    - Efficient untuk mobile dan embedded devices
    - Menggunakan depthwise separable convolutions
    - Inverted residual structure dengan linear bottlenecks
    - Parameter lebih sedikit dibanding model tradisional
    """)