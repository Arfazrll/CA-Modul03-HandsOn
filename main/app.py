import streamlit as st

st.set_page_config(
    page_title="Deep Learning Image Classifier",
    layout="wide"
)

st.title("Deep Learning Image Classifier")
st.write("Platform untuk memahami cara kerja CNN dan Transfer Learning dalam klasifikasi gambar")

st.markdown("---")

st.header("Deep Learning visualization")

st.write("""
dirancang untuk membantu memahami konsep Deep Learning melalui visualisasi interaktif.
Anda dapat melihat bagaimana model neural network memproses gambar dan membuat prediksi. semoga mengerti ya adik adik hehe
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CNN Classification")
    st.write("**Dataset:** Rock Paper Scissors")
    st.write("**Kelas:** Rock, Paper, Scissors")
    st.write("**Ukuran Input:** 150x150 pixels")
    st.write("""
    Pada halaman ini Anda akan belajar:
    - Bagaimana CNN memproses gambar
    - Cara kerja convolution dan pooling
    - Proses ekstraksi fitur
    - Prediksi berdasarkan feature maps
    """)

with col2:
    st.subheader("MobileNetV2 Transfer Learning")
    st.write("**Dataset:** Hyena vs Cheetah")
    st.write("**Kelas:** Hyena, Cheetah")
    st.write("**Ukuran Input:** 224x224 pixels")
    st.write("""
    Pada halaman ini Anda akan belajar:
    - Konsep Transfer Learning
    - Pre-trained model ImageNet
    - Feature extraction process
    - Fine-tuning untuk task baru
    """)

st.markdown("---")

st.info("Pilih salah satu model dari sidebar untuk memulai")