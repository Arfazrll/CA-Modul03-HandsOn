import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="MobileNetV2 Transfer Learning", layout="wide")

st.title("Transfer Learning: MobileNetV2")
st.write("Memahami bagaimana Transfer Learning menggunakan model pre-trained untuk task baru")

@st.cache_resource
def load_model():
    model_path = 'best_transfer_model.keras'
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            st.success("Model MobileNetV2 berhasil di-load!")
            return model, True
        except Exception as e:
            st.warning(f"Model file ada tapi gagal di-load: {str(e)}")
    
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
    
    st.error("""
    ⚠️ MODEL BELUM DI-TRAIN UNTUK HYENA VS CHEETAH!
    
    Model ini hanya menggunakan pre-trained ImageNet weights tanpa fine-tuning untuk Hyena/Cheetah.
    
    Untuk prediksi yang akurat:
    1. Train model di Colab menggunakan notebook MobileNetv2.ipynb
    2. Download file 'best_transfer_model.keras' yang di-save otomatis
    3. Upload file tersebut ke folder aplikasi ini
    """)
    
    return model, False

def preprocess_mobilenetv2(image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array.copy())
    return img_resized, img_array, img_preprocessed

model, is_trained = load_model()
class_names = ['Cheetah', 'Hyena']

st.markdown("---")

st.header("Apa itu Transfer Learning?")
st.write("""
**Transfer Learning** adalah teknik machine learning dimana kita menggunakan model yang sudah dilatih pada dataset besar 
(seperti ImageNet dengan 14 juta gambar) dan mengadaptasinya untuk task baru dengan dataset lebih kecil.

**Keuntungan:**
- Tidak perlu melatih model dari nol
- Lebih cepat dan efisien
- Bekerja baik dengan dataset kecil
- Memanfaatkan knowledge yang sudah dipelajari
""")

st.markdown("---")

st.header("MobileNetV2 Pre-trained pada ImageNet")
st.write("""
**MobileNetV2** adalah model yang sudah dilatih untuk mengenali 1000 kategori objek dari dataset ImageNet.
Model ini telah belajar mengenali:
- Fitur dasar: Garis, tepi, warna, tekstur
- Fitur menengah: Bentuk, pola
- Fitur kompleks: Bagian objek, kombinasi bentuk

Kita akan menggunakan knowledge ini untuk membedakan Cheetah dan Hyena.
""")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Layer yang Di-freeze (Tidak Dilatih):**
    - Base MobileNetV2
    - Sudah bisa ekstrak fitur umum
    - Parameter: 2.2 juta
    """)

with col2:
    if is_trained:
        st.success("""
        **Layer yang Dilatih:**
        - Global Average Pooling
        - Dense Layer (128 neurons)
        - Output Layer (2 classes)
        - Model SUDAH dilatih!
        """)
    else:
        st.warning("""
        **Layer yang Perlu Dilatih:**
        - Global Average Pooling
        - Dense Layer (128 neurons)
        - Output Layer (2 classes)
        - ⚠️ Model BELUM dilatih!
        """)

st.markdown("---")

if not is_trained:
    st.warning("""
    ### Cara Mendapatkan Model yang Sudah Di-train:
    
    1. Buka notebook **MobileNetv2.ipynb** di Google Colab
    2. Jalankan semua cell sampai training selesai
    3. File 'best_transfer_model.keras' akan otomatis ter-save
    4. Download file tersebut dengan menambahkan cell:
    ```python
    from google.colab import files
    files.download('best_transfer_model.keras')
    ```
    5. Upload file ke folder tempat app.py berada
    6. Restart aplikasi Streamlit
    """)
    
    st.info("""
    **Untuk saat ini, aplikasi tetap bisa digunakan untuk mempelajari:**
    - Transfer Learning concept
    - RGB analysis dan preprocessing
    - Feature extraction dari MobileNetV2
    - Arsitektur model
    
    Namun prediksi akan tidak akurat karena model belum di-fine-tune untuk Hyena/Cheetah.
    """)

st.markdown("---")

st.header("Upload Gambar untuk Prediksi")
uploaded_file = st.file_uploader("Pilih gambar Cheetah atau Hyena", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Gambar Setelah Preprocessing")
        img_resized, img_array, img_preprocessed = preprocess_mobilenetv2(image)
        st.image(img_resized, use_column_width=True)
        st.caption("Ukuran: 224x224 pixels")
        st.caption(f"Original range: [0, 255]")
        st.caption(f"Preprocessed range: [{img_preprocessed.min():.2f}, {img_preprocessed.max():.2f}]")
    
    st.markdown("---")
    
    st.header("Analisis RGB Channels")
    st.write("""
    MobileNetV2 menerima input 3 channel (RGB). Mari lihat distribusi intensitas warna pada gambar Anda.
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
    
    st.header("Distribusi Intensitas Pixel")
    
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
        xaxis_title="Intensitas Pixel (0-255)",
        yaxis_title="Frekuensi",
        height=400
    )
    
    st.plotly_chart(fig_hist, use_column_width=True)
    st.caption("Histogram menunjukkan distribusi intensitas untuk setiap channel warna")
    
    st.markdown("---")
    
    st.header("Prediksi dan Klasifikasi")
    
    if is_trained:
        button_label = "Jalankan Prediksi dengan MobileNetV2"
    else:
        button_label = "Coba Prediksi (Model Belum Di-train untuk Hyena/Cheetah)"
    
    if st.button(button_label, type="primary"):
        with st.spinner("Memproses gambar melalui MobileNetV2..."):
            img_input = np.expand_dims(img_preprocessed, axis=0)
            predictions = model.predict(img_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
        
        if is_trained:
            st.success(f"Prediksi: **{predicted_class}** dengan confidence **{confidence:.2f}%**")
        else:
            st.warning(f"Prediksi (TIDAK AKURAT): **{predicted_class}** - Confidence: {confidence:.2f}%")
            st.error("⚠️ Prediksi ini TIDAK AKURAT karena model belum di-fine-tune untuk Hyena vs Cheetah!")
        
        st.subheader("Distribusi Probabilitas")
        if is_trained:
            st.write("""
            Model memberikan probabilitas untuk setiap kelas. Output layer menggunakan aktivasi softmax 
            yang mengubah nilai menjadi probabilitas (total 100%).
            """)
        else:
            st.write("⚠️ Probabilitas ini TIDAK RELIABLE karena model belum dilatih untuk membedakan Hyena dan Cheetah!")
        
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
            xaxis_title="Kelas",
            yaxis_title="Probabilitas (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_pred, use_column_width=True)
        
        st.markdown("---")
        
        st.header("Feature Extraction dari MobileNetV2")
        st.write("""
        MobileNetV2 mengekstrak 1280 fitur dari gambar input. Fitur-fitur ini merepresentasikan 
        karakteristik penting yang dipelajari dari ImageNet. Mari lihat fitur-fitur yang paling penting.
        """)
        
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-3].output
        )
        features = feature_extractor.predict(img_input, verbose=0)
        
        st.write(f"**Shape fitur yang diekstrak:** {features.shape}")
        st.caption("(1, 1280) berarti 1280 fitur untuk 1 gambar")
        
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
            title="Top 30 Fitur Terpenting (Berdasarkan Magnitude)",
            xaxis_title="Index Fitur",
            yaxis_title="Nilai Absolut",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_feat, use_column_width=True)
        
        if is_trained:
            st.write("""
            Fitur dengan magnitude besar (positif atau negatif) memiliki pengaruh lebih kuat 
            dalam menentukan klasifikasi akhir antara Cheetah dan Hyena.
            """)
        else:
            st.warning("""
            ⚠️ Fitur-fitur ini adalah hasil dari ImageNet pre-training.
            Setelah fine-tuning, fitur-fitur ini akan disesuaikan untuk membedakan Cheetah dan Hyena.
            """)
        
        st.markdown("---")
        
        st.header("Proses Transfer Learning")
        st.write("""
        Berikut adalah alur lengkap bagaimana gambar Anda diproses:
        
        **Langkah 1: Input Processing**
        - Gambar diubah ke 224x224 pixels
        - Preprocessing khusus MobileNetV2 (bukan hanya /255!)
        - Normalisasi: mean subtraction dan scaling
        
        **Langkah 2: Feature Extraction (MobileNetV2)**
        - Gambar melewati 53 convolutional layers
        - Menghasilkan feature maps 7x7x1280
        - Mengekstrak fitur dari simple ke complex
        - Menggunakan depthwise separable convolutions
        
        **Langkah 3: Global Average Pooling**
        - Feature maps 7x7x1280 dirata-ratakan
        - Menghasilkan vector 1280 fitur
        - Mengurangi overfitting
        
        **Langkah 4: Dense Classification**
        - 1280 fitur → 128 neurons (dengan ReLU)
        - Dropout 50% untuk regularisasi
        - 128 neurons → 2 outputs (Cheetah, Hyena)
        
        **Langkah 5: Softmax**
        - Mengubah output menjadi probabilitas
        - Prediksi = kelas dengan probabilitas tertinggi
        """)
        
        st.markdown("---")
        
        st.header("Visualisasi Arsitektur")
        
        layers_info = [
            {"name": "Input Image", "size": "224x224x3", "params": "0", "desc": "RGB Image with MobileNetV2 preprocessing"},
            {"name": "MobileNetV2 Base", "size": "7x7x1280", "params": "2.2M", "desc": "Feature Extraction (frozen)"},
            {"name": "Global Avg Pooling", "size": "1280", "params": "0", "desc": "Spatial dimensions reduction"},
            {"name": "Dense + ReLU", "size": "128", "params": "163K", "desc": "Feature combination"},
            {"name": "Dropout (0.5)", "size": "128", "params": "0", "desc": "Regularization"},
            {"name": "Output + Softmax", "size": "2", "params": "258", "desc": "Classification (Cheetah, Hyena)"}
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
        
        st.header("Kenapa MobileNetV2 Efektif?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Depthwise Separable Convolution")
            st.write("""
            MobileNetV2 menggunakan teknik khusus yang memisahkan:
            - **Depthwise Convolution**: Proses setiap channel secara terpisah
            - **Pointwise Convolution**: Kombinasikan channel dengan 1x1 filter
            
            Hasil: 8-9x lebih sedikit parameter dibanding CNN biasa
            """)
        
        with col2:
            st.subheader("Inverted Residual Blocks")
            st.write("""
            Struktur unik yang:
            - Ekspansi fitur dengan 1x1 conv
            - Depthwise 3x3 conv untuk spatial features
            - Proyeksi linear untuk output
            
            Hasil: Efisien untuk mobile dan embedded devices
            """)
        
        st.markdown("---")
        

else:
    st.info("Silakan upload gambar Cheetah atau Hyena untuk memulai prediksi dan melihat visualisasi")
    
    st.markdown("---")
    
    st.header("Perbandingan: CNN vs Transfer Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CNN dari Nol")
        st.write("""
        **Keuntungan:**
        - Kontrol penuh atas arsitektur
        - Dioptimasi untuk dataset spesifik
        
        **Kekurangan:**
        - Butuh dataset besar (>10,000 gambar)
        - Training lama (berjam-jam sampai berhari-hari)
        - Butuh resource komputasi besar
        - Rawan overfitting pada dataset kecil
        """)
    
    with col2:
        st.subheader("Transfer Learning")
        st.write("""
        **Keuntungan:**
        - Bekerja dengan dataset kecil (<1,000 gambar)
        - Training cepat (menit sampai jam)
        - Resource komputasi lebih sedikit
        - Akurasi tinggi dengan data minimal
        
        **Kekurangan:**
        - Terbatas pada domain yang mirip
        - Ukuran model lebih besar
        """)
    
    st.markdown("---")
    
    st.header("Dataset: Cheetah vs Hyena")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cheetah")
        st.write("""
        Karakteristik visual:
        - Bintik hitam bulat kecil
        - Garis hitam dari mata ke mulut (tear mark)
        - Tubuh lebih ramping dan atletis
        - Warna coklat keemasan terang
        """)
    
    with col2:
        st.subheader("Hyena")
        st.write("""
        Karakteristik visual:
        - Bintik tidak teratur atau garis
        - Tidak ada tear mark
        - Tubuh lebih besar dan kokoh
        - Warna lebih gelap, coklat keabuan
        """)
    
    st.write("""
    Model MobileNetV2 yang sudah dilatih dapat mengenali perbedaan subtle ini 
    karena sudah memiliki knowledge tentang tekstur, pola, dan bentuk dari training ImageNet.
    """)