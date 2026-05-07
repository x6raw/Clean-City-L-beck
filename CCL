import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CleanCity Lübeck",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------
# CSS DESIGN
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

.title {
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle {
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

.metric {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.metric-number {
    font-size:40px;
    color:#38bdf8;
    font-weight:bold;
}

.metric-label {
    color:#cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="title">🌍 CleanCity Lübeck</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">KI-gestützte Müll- & Objekterkennung mit YOLOv8</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.header("⚙️ Einstellungen")

    confidence = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25,
        0.05
    )

    st.markdown("---")

    st.info("""
    📌 Diese KI erkennt:
    
    - Personen
    - Flaschen
    - Taschen
    - Autos
    - Fahrräder
    - Müllähnliche Objekte
    """)

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# DETECTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Originalbild")
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bild zu numpy
    img_array = np.array(image)

    # KI Analyse
    with st.spinner("🧠 KI analysiert das Bild..."):
        time.sleep(1)

        results = model.predict(
            source=img_array,
            conf=confidence,
            save=False
        )

    # Ergebnisbild
    result_img = results[0].plot()

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 KI-Erkennung")
        st.image(result_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # STATS
    # ---------------------------------------------------
    boxes = results[0].boxes

    total_objects = len(boxes)

    avg_conf = 0

    if total_objects > 0:
        avg_conf = np.mean([
            float(box.conf[0]) for box in boxes
        ])

    st.markdown("## 📊 Analyse")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{total_objects}</div>
            <div class="metric-label">Objekte erkannt</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{avg_conf:.2f}</div>
            <div class="metric-label">Ø Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    unique_classes = len(set([
        int(box.cls[0]) for box in boxes
    ]))

    with c3:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{unique_classes}</div>
            <div class="metric-label">Klassen erkannt</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # OBJECT TABLE
    # ---------------------------------------------------
    st.markdown("## 📋 Erkannte Objekte")

    data = []

    for box in boxes:

        cls_id = int(box.cls[0])

        conf = float(box.conf[0])

        data.append({
            "Objekt": model.names[cls_id],
            "Confidence": round(conf, 2)
        })

    if len(data) > 0:

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Keine Objekte erkannt.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    "<center>🌍 CleanCity Lübeck | KI-Projekt mit Streamlit & YOLOv8</center>",
    unsafe_allow_html=True
)
