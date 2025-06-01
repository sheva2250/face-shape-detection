import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import torch.nn as nn

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, 5)
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Class name and detail
class_info = {
    "Diamond 💎": {
        "desc": "Wajah berbentuk diamond memiliki bagian dagu dan dahi sempit dengan tulang pipi lebar.",
        "tips": "Gunakan gaya rambut yang menambah volume di dahi atau bagian bawah wajah, seperti bob atau wave panjang."
    },
    "Heart ❤️": {
        "desc": "Bentuk wajah hati memiliki dahi lebar dan dagu yang runcing seperti segitiga terbalik.",
        "tips": "Cocok dengan poni samping atau rambut panjang dengan layer."
    },
    "Oblong 🧊": {
        "desc": "Wajah oblong cenderung panjang dengan garis pipi lurus dan dagu berbentuk persegi.",
        "tips": "Hindari rambut terlalu panjang lurus, coba gaya wave atau poni samping untuk keseimbangan."
    },
    "Oval 🥚": {
        "desc": "Wajah oval memiliki proporsi seimbang dan sedikit lebih panjang daripada lebar.",
        "tips": "Beruntung! Hampir semua gaya rambut cocok dengan bentuk wajah ini."
    },
    "Round ⚪": {
        "desc": "Wajah bulat ditandai dengan panjang dan lebar hampir sama, serta pipi penuh.",
        "tips": "Gunakan gaya yang menambah tinggi seperti layer panjang atau volume di atas kepala."
    }
}

# Mapping index to class
idx_to_class = list(class_info.keys())

# --- UI DESIGN ---

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>✨ AI Prediksi Bentuk Wajah</h1>
    <p style='text-align: center; font-size: 18px;'>
        Gunakan kamera untuk mengenali bentuk wajah dan dapatkan tips gaya rambut terbaik!
    </p>
    <hr>
""", unsafe_allow_html=True)

camera_image = st.camera_input("📷 Ambil gambar wajah dengan pencahayaan cukup:")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Gambar Anda", use_container_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        pred_label = idx_to_class[pred.item()]
        detail = class_info[pred_label]

    with col2:
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f1f8e9;'>
                <h2 style='color: #33691E;'>🔍 Prediksi: {pred_label}</h2>
                <p><b>Deskripsi:</b> {detail['desc']}</p>
                <p><b>Rekomendasi Gaya Rambut:</b> 💇‍♀️ {detail['tips']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.success("Prediksi berhasil! Anda bisa coba ambil gambar lain untuk mengecek kembali.")

    # Optional fun fact
    with st.expander("🤖 Fakta AI!"):
        st.markdown("""
            - Model yang digunakan adalah **EfficientNet B4**, dilatih dengan data bentuk wajah.
            - AI mengenali pola berdasarkan **proporsi wajah**, **struktur tulang**, dan **kontur garis**.
            - Gunakan AI ini sebagai alat bantu, bukan penentu tunggal—setiap wajah unik!
        """)

# Footer
st.markdown("<hr><center><small>© 2025 - Aplikasi Prediksi Wajah dengan AI</small></center>", unsafe_allow_html=True)
