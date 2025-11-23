import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import convolve
import os
import requests

st.set_page_config(page_title="CNN Classification", layout="wide")

st.title("CNN Classification: Rock Paper Scissors")
st.write("Memahami bagaimana Convolutional Neural Network memproses dan mengklasifikasi gambar")

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
    model_url = "https://github.com/Arfazrll/CA-Modul03-HandsOn/releases/download/modelresultCNN/rock_paper_scissors_model.h5"
    model_path = 'rock_paper_scissors_model.h5'
    
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
        st.success("Model loaded successfully")
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
    st.warning("Using untrained model architecture. Predictions will be random.")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def preprocess_image(image):
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    return img_resized, img_array

def apply_convolution(image, kernel):
    img_gray = np.array(image.convert('L'))
    convolved = convolve(img_gray, kernel, mode='constant')
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    return convolved

model, is_trained = load_model()
class_names = ['paper', 'rock', 'scissors']

st.markdown("---")

if is_trained:
    st.success("Model ready for accurate predictions")
else:
    st.warning("Model not trained. Predictions will be random for educational purposes only.")

st.markdown("---")

st.header("Step 1: Upload Image")
uploaded_file = st.file_uploader("Choose Rock, Paper, or Scissors image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("After Preprocessing")
        img_resized, img_array = preprocess_image(image)
        st.image(img_resized, use_column_width=True)
        st.caption(f"Size: 150x150 | Normalized: 0-1 | Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    st.markdown("---")
    
    st.header("Step 2: Understanding Input - RGB Channels")
    st.write("Images consist of 3 color channels: Red, Green, and Blue. Each channel is a matrix with intensity values 0-255 (normalized to 0-1). CNN processes all three channels simultaneously to extract features.")
    
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
    
    with col2:
        st.subheader("Green Channel")
        fig_g = go.Figure(data=go.Heatmap(
            z=img_display[:,:,1][::-1],
            colorscale='Greens',
            showscale=True
        ))
        fig_g.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_g, use_column_width=True)
    
    with col3:
        st.subheader("Blue Channel")
        fig_b = go.Figure(data=go.Heatmap(
            z=img_display[:,:,2][::-1],
            colorscale='Blues',
            showscale=True
        ))
        fig_b.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_b, use_column_width=True)
    
    st.markdown("---")
    
    st.header("Step 3: Convolution Operation")
    st.write("Convolution is a mathematical operation that uses filters (kernels) to extract features from images. Filters slide across the image and perform element-wise multiplication, then sum the results. This helps detect patterns like lines, edges, textures, and shapes.")
    
    kernel_options = {
        'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Blur': np.ones((3, 3)) / 9,
        'Vertical Edge': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Horizontal Edge': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    }
    
    selected_kernel = st.selectbox("Select filter to see its effect:", list(kernel_options.keys()))
    kernel = kernel_options[selected_kernel]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kernel Matrix (3x3)")
        fig_kernel = go.Figure(data=go.Heatmap(
            z=kernel,
            colorscale='RdBu',
            text=kernel,
            texttemplate='%{text:.2f}',
            textfont={"size": 14},
            showscale=False
        ))
        fig_kernel.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_kernel, use_column_width=True)
        st.caption("3x3 filter that slides across the image")
    
    with col2:
        st.subheader("Convolution Result")
        convolved = apply_convolution(img_resized, kernel)
        st.image(convolved, use_column_width=True)
        st.caption("Image after processing with selected filter")
    
    st.markdown("---")
    
    st.header("Step 4: Prediction and Classification")
    
    button_label = "Run Prediction" if is_trained else "Try Prediction (Untrained Model)"
    
    if st.button(button_label, type="primary"):
        with st.spinner("Processing image through CNN..."):
            img_input = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
        
        if is_trained:
            st.success(f"Prediction: **{predicted_class.upper()}** with confidence **{confidence:.2f}%**")
        else:
            st.warning(f"Prediction (RANDOM): **{predicted_class.upper()}** - Confidence: {confidence:.2f}%")
            st.error("This prediction is NOT ACCURATE because the model is untrained")
        
        st.subheader("Probability Distribution")
        st.write("The model gives probabilities for each class. The class with highest probability is the prediction." if is_trained else "These probabilities are RANDOM because the model hasn't learned anything")
        
        fig_pred = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=predictions[0] * 100,
                marker_color=['#3498db', '#e74c3c', '#2ecc71'],
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
        
        st.header("Step 5: Feature Maps Visualization")
        st.write("Feature maps are outputs from each convolution layer. They show what features the model detects. Early layers detect simple features (lines, edges), deeper layers detect complex features (shapes, patterns).")
        
        layer_outputs = []
        layer_names = []
        
        temp_input = tf.keras.Input(shape=(150, 150, 3))
        temp_x = temp_input
        
        for layer in model.layers[:3]:
            temp_x = layer(temp_x)
            if 'conv' in layer.name:
                layer_outputs.append(temp_x)
                layer_names.append(layer.name)
        
        feature_model = tf.keras.Model(inputs=temp_input, outputs=layer_outputs)
        features = feature_model.predict(img_input, verbose=0)
        
        for idx, (feature_map, name) in enumerate(zip(features, layer_names)):
            st.subheader(f"Layer: {name}")
            st.caption(f"Shape: {feature_map.shape}")
            
            num_filters = min(8, feature_map.shape[-1])
            cols = st.columns(4)
            
            for i in range(num_filters):
                with cols[i % 4]:
                    fmap = feature_map[0, :, :, i]
                    fig_fm = go.Figure(data=go.Heatmap(
                        z=fmap,
                        colorscale='Viridis',
                        showscale=False
                    ))
                    fig_fm.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=20, b=0),
                        title=f"Filter {i+1}"
                    )
                    st.plotly_chart(fig_fm, use_column_width=True)
        
        if not is_trained:
            st.warning("These feature maps are also random because the model is untrained")
        
        st.markdown("---")
        
        st.header("CNN Architecture")
        st.write("""
        Complete structure of the CNN used:
        
        1. Input Layer: 150x150x3 (RGB image)
        2. Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU activation
        3. MaxPooling2D: 2x2, reduces spatial dimensions
        4. Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU activation
        5. MaxPooling2D: 2x2, reduces spatial dimensions
        6. Conv2D Layer 3: 128 filters, 3x3 kernel, ReLU activation
        7. MaxPooling2D: 2x2, reduces spatial dimensions
        8. Flatten: Convert 3D to 1D vector
        9. Dropout: 0.5 rate for regularization
        10. Dense Layer: 512 neurons, ReLU activation
        11. Output Layer: 3 neurons (Rock, Paper, Scissors), Softmax activation
        """)
        
        st.info("""
        How CNN Works:
        - Convolution: Extracts local features from images
        - Pooling: Reduces dimensions and increases invariance
        - Activation (ReLU): Adds non-linearity
        - Flatten: Converts feature maps to vector
        - Dense: Performs classification based on extracted features
        - Softmax: Converts output to probabilities (sum=100%)
        """)

else:
    st.info("Please upload an image to start classification and see visualizations")
    
    st.markdown("---")
    
    st.header("About CNN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is CNN?")
        st.write("Convolutional Neural Network (CNN) is a type of neural network designed specifically for processing grid-like data such as images. CNN consists of several layers: Convolutional Layer (extracts features), Pooling Layer (reduces dimensions), and Fully Connected Layer (performs classification).")
    
    with col2:
        st.subheader("Why is CNN Effective?")
        st.write("CNN is effective for vision tasks because of: Parameter Sharing (same filter used across the image), Spatial Hierarchy (detects features from simple to complex), and Translation Invariance (can recognize objects at different positions).")
    
    st.markdown("---")
    
    st.subheader("Dataset: Rock Paper Scissors")
    st.write("This model is trained to classify 3 hand gestures: Rock (clenched fist), Paper (open hand), and Scissors (two fingers extended). The model uses a custom CNN architecture with 3 convolutional layers and can distinguish between these three classes accurately after training.")

def preprocess_image(image):
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    return img_resized, img_array

def apply_convolution(image, kernel):
    img_gray = np.array(image.convert('L'))
    convolved = convolve(img_gray, kernel, mode='constant')
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    return convolved

model, is_trained = load_model()
class_names = ['paper', 'rock', 'scissors']

st.markdown("---")

if is_trained:
    st.success("Model sudah siap digunakan! Model ini sudah di-training dan siap untuk prediksi akurat.")
else:
    st.warning("""
    Model belum tersedia atau gagal di-load. 
    Aplikasi akan tetap berfungsi untuk pembelajaran visualisasi, namun prediksi akan random.
    """)

st.markdown("---")

st.header("Langkah 1: Upload Gambar")
uploaded_file = st.file_uploader("Pilih gambar Rock, Paper, atau Scissors", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Gambar Setelah Preprocessing")
        img_resized, img_array = preprocess_image(image)
        st.image(img_resized, use_column_width=True)
        st.caption(f"Ukuran: 150x150 pixels | Normalisasi: 0-1 | Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    st.markdown("---")
    
    st.header("Langkah 2: Memahami Input - RGB Channels")
    st.write("""
    Gambar digital terdiri dari 3 channel warna: **Red (Merah)**, **Green (Hijau)**, dan **Blue (Biru)**.
    Setiap channel adalah matrix dengan nilai intensitas 0-255 (setelah normalisasi menjadi 0-1).
    CNN memproses ketiga channel ini secara bersamaan untuk mengekstrak fitur.
    """)
    
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
    
    with col2:
        st.subheader("Green Channel")
        fig_g = go.Figure(data=go.Heatmap(
            z=img_display[:,:,1][::-1],
            colorscale='Greens',
            showscale=True
        ))
        fig_g.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_g, use_column_width=True)
    
    with col3:
        st.subheader("Blue Channel")
        fig_b = go.Figure(data=go.Heatmap(
            z=img_display[:,:,2][::-1],
            colorscale='Blues',
            showscale=True
        ))
        fig_b.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_b, use_column_width=True)
    
    st.markdown("---")
    
    st.header("Langkah 3: Convolution Operation")
    st.write("""
    **Convolution** adalah operasi matematika yang menggunakan filter (kernel) untuk mengekstrak fitur dari gambar.
    Filter bergerak melintasi gambar dan melakukan operasi perkalian element-wise, kemudian menjumlahkan hasilnya.
    Ini membantu mendeteksi pola seperti garis, tepi, tekstur, dan bentuk.
    """)
    
    kernel_options = {
        'Edge Detection (Deteksi Tepi)': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Sharpen (Mempertajam)': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Blur (Mengaburkan)': np.ones((3, 3)) / 9,
        'Vertical Edge (Tepi Vertikal)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Horizontal Edge (Tepi Horizontal)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    }
    
    selected_kernel = st.selectbox("Pilih jenis filter untuk melihat efeknya:", list(kernel_options.keys()))
    kernel = kernel_options[selected_kernel]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kernel/Filter Matrix (3x3)")
        fig_kernel = go.Figure(data=go.Heatmap(
            z=kernel,
            colorscale='RdBu',
            text=kernel,
            texttemplate='%{text:.2f}',
            textfont={"size": 14},
            showscale=False
        ))
        fig_kernel.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_kernel, use_column_width=True)
        st.caption("Filter 3x3 yang akan bergerak melintasi gambar")
    
    with col2:
        st.subheader("Hasil Convolution")
        convolved = apply_convolution(img_resized, kernel)
        st.image(convolved, use_column_width=True)
        st.caption("Gambar setelah diproses dengan filter yang dipilih")
    
    st.markdown("---")
    
    st.header("Langkah 4: Prediksi dan Klasifikasi")
    
    if is_trained:
        button_label = "Jalankan Prediksi"
    else:
        button_label = "Coba Prediksi (Model Belum Tersedia)"
    
    if st.button(button_label, type="primary"):
        with st.spinner("Memproses gambar melalui CNN..."):
            img_input = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
        
        if is_trained:
            st.success(f"Prediksi: **{predicted_class.upper()}** dengan confidence **{confidence:.2f}%**")
        else:
            st.warning(f"Prediksi (RANDOM): **{predicted_class.upper()}** - Confidence: {confidence:.2f}%")
            st.error("⚠️ Prediksi ini TIDAK AKURAT karena model belum tersedia atau gagal di-load!")
        
        st.subheader("Distribusi Probabilitas")
        if is_trained:
            st.write("Model memberikan probabilitas untuk setiap kelas. Kelas dengan probabilitas tertinggi adalah hasil prediksi.")
        else:
            st.write("⚠️ Probabilitas ini RANDOM karena model belum pernah belajar apa-apa!")
        
        fig_pred = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=predictions[0] * 100,
                marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                text=[f'{val:.2f}%' for val in predictions[0] * 100],
                textposition='auto'
            )
        ])
        fig_pred.update_layout(
            xaxis_title="Kelas",
            yaxis_title="Probabilitas (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_pred, use_column_width=True)
        
        st.markdown("---")
        
        st.header("Langkah 5: Visualisasi Feature Maps")
        st.write("""
        Feature maps adalah output dari setiap layer convolution. Mereka menunjukkan fitur apa yang dideteksi oleh model.
        Layer awal mendeteksi fitur sederhana (garis, tepi), layer dalam mendeteksi fitur kompleks (bentuk, pola).
        """)
        
        layer_outputs = []
        layer_names = []
        
        temp_input = tf.keras.Input(shape=(150, 150, 3))
        temp_x = temp_input
        
        for layer in model.layers[:3]:
            temp_x = layer(temp_x)
            if 'conv' in layer.name:
                layer_outputs.append(temp_x)
                layer_names.append(layer.name)
        
        feature_model = tf.keras.Model(inputs=temp_input, outputs=layer_outputs)
        features = feature_model.predict(img_input, verbose=0)
        
        for idx, (feature_map, name) in enumerate(zip(features, layer_names)):
            st.subheader(f"Layer: {name}")
            st.caption(f"Shape: {feature_map.shape}")
            
            num_filters = min(8, feature_map.shape[-1])
            cols = st.columns(4)
            
            for i in range(num_filters):
                with cols[i % 4]:
                    fmap = feature_map[0, :, :, i]
                    fig_fm = go.Figure(data=go.Heatmap(
                        z=fmap,
                        colorscale='Viridis',
                        showscale=False
                    ))
                    fig_fm.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=20, b=0),
                        title=f"Filter {i+1}"
                    )
                    st.plotly_chart(fig_fm, use_column_width=True)
        
        if not is_trained:
            st.warning("⚠️ Feature maps ini juga random karena model belum di-train.")
        
        st.markdown("---")
        
        st.header("Arsitektur CNN")
        st.write("""
        Berikut adalah struktur lengkap CNN yang digunakan:
        
        1. **Input Layer**: 150x150x3 (RGB image)
        2. **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
        3. **MaxPooling2D**: 2x2, reduces spatial dimensions
        4. **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
        5. **MaxPooling2D**: 2x2, reduces spatial dimensions
        6. **Conv2D Layer 3**: 128 filters, 3x3 kernel, ReLU activation
        7. **MaxPooling2D**: 2x2, reduces spatial dimensions
        8. **Flatten**: Convert 3D to 1D vector
        9. **Dropout**: 0.5 rate for regularization
        10. **Dense Layer**: 512 neurons, ReLU activation
        11. **Output Layer**: 3 neurons (Rock, Paper, Scissors), Softmax activation
        """)
        
        st.info("""
        **Cara Kerja CNN:**
        - **Convolution**: Mengekstrak fitur lokal dari gambar
        - **Pooling**: Mengurangi dimensi dan meningkatkan invariance
        - **Activation (ReLU)**: Menambahkan non-linearity
        - **Flatten**: Mengubah feature maps menjadi vector
        - **Dense**: Melakukan klasifikasi berdasarkan fitur yang diekstrak
        - **Softmax**: Mengubah output menjadi probabilitas (sum=100%)
        """)

else:
    st.info("Silakan upload gambar untuk memulai proses klasifikasi dan melihat visualisasi")
    
    st.markdown("---")
    
    st.header("Tentang CNN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Apa itu CNN?")
        st.write("""
        **Convolutional Neural Network (CNN)** adalah jenis neural network yang dirancang khusus untuk memproses data grid seperti gambar.
        
        CNN terdiri dari beberapa layer:
        - **Convolutional Layer**: Mengekstrak fitur dari gambar
        - **Pooling Layer**: Mengurangi dimensi dan komputasi
        - **Fully Connected Layer**: Melakukan klasifikasi
        """)
    
    with col2:
        st.subheader("Mengapa CNN Efektif?")
        st.write("""
        CNN efektif untuk vision tasks karena:
        - **Parameter Sharing**: Filter yang sama digunakan di seluruh gambar
        - **Spatial Hierarchy**: Mendeteksi fitur dari sederhana ke kompleks
        - **Translation Invariance**: Dapat mengenali objek di posisi berbeda
        """)
    
    st.markdown("---")
    
    st.subheader("Dataset: Rock Paper Scissors")
    st.write("""
    Model ini dilatih untuk mengklasifikasi 3 gesture tangan:
    - **Rock (Batu)**: Tangan mengepal
    - **Paper (Kertas)**: Tangan terbuka
    - **Scissors (Gunting)**: Dua jari terentang
    
    Model menggunakan arsitektur CNN custom dengan 3 convolutional layers dan dapat membedakan ketiga kelas dengan akurat setelah di-train.
    """)