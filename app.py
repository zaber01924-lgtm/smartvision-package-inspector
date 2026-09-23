
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page settings
st.set_page_config(
    page_title="SmartVision",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 SmartVision")
st.subheader("AI Package Quality Inspector")

st.write(
    "Take a photo of a package and let the AI classify it as "
    "**Good** or **Broken**."
)

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Camera
photo = st.camera_input("📷 Take a picture of the package")

if photo is not None:

    image = Image.open(photo)

    st.image(
        image,
        caption="Captured package",
        use_container_width=True
    )

    # Run AI prediction
    results = model.predict(
        source=image,
        imgsz=224,
        verbose=False
    )

    result = results[0]

    predicted_class = result.probs.top1
    confidence = float(result.probs.top1conf)
    class_name = result.names[predicted_class]

    st.divider()

    if class_name == "good":
        st.success("✅ GOOD PACKAGE")
    else:
        st.error("⚠️ BROKEN PACKAGE")

    st.metric(
        "AI Confidence",
        f"{confidence:.1%}"
    )
