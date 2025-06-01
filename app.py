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

# Mapping indeks ke nama kelas
idx_to_class = {
    0: "Diamond 💎", 1: "Heart ❤️", 2: "Oblong 🧊", 3: "Oval 🥚", 4: "Round ⚪"
}

# --- UI DESIGN ---

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📸 Deteksi Bentuk Wajah</h1>
    <p style='text-align: center; font-size: 16px;'>
        Ambil gambar wajah Anda dan temukan jenis bentuk wajah secara otomatis menggunakan AI!
    </p>
    <hr>
""", unsafe_allow_html=True)

camera_image = st.camera_input("Ambil gambar wajah dengan pencahayaan yang cukup:")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="📷 Gambar yang Diambil", use_container_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        pred_label = idx_to_class[pred.item()]

    with col2:
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #e0f7fa; text-align: center;'>
                <h3 style='color: #00796B;'>🔍 Hasil Prediksi:</h3>
                <h1 style='color: #004D40;'>{pred_label}</h1>
            </div>
        """, unsafe_allow_html=True)

    st.info("Gunakan kembali kamera untuk mengambil gambar lain jika diperlukan.")
