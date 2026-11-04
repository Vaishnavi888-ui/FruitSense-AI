"""
FruitSense AI - AI-Powered Mango Maturity Assessment Application
Streamlit MVP for 24-Hour Hackathon
Theme: Green & Cream Agricultural Dashboard
"""

import os
import io
import datetime
import pandas as pd
from PIL import Image
import streamlit as st

from predictor import (
    check_model_status,
    assess_mango,
    get_stage_guidance,
    CATEGORIES,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FruitSense AI | Mango Maturity Assessment",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Green & Cream Agricultural Theme CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Background & Accent Highlights */
    .stApp {
        background-color: #FAF7F2;
        color: #1B4332;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
        color: #FAF7F2;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(27, 67, 50, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #FAF7F2;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        color: #D8F3DC;
        margin-bottom: 0px;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .status-pill-prototype {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .status-pill-connected {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }

    /* Card Containers */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1B4332;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Stage Badges */
    .stage-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .stage-unripe {
        background-color: #D8F3DC;
        color: #1B4332;
        border: 2px solid #52B788;
    }
    .stage-semiripe {
        background-color: #FEF3C7;
        color: #B45309;
        border: 2px solid #F59E0B;
    }
    .stage-ripe {
        background-color: #FFEDD5;
        color: #C2410C;
        border: 2px solid #F97316;
    }

    /* Guidance Bullet Pill */
    .insight-row {
        background-color: #F8F6F0;
        border-left: 4px solid #2D6A4F;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .insight-label {
        font-weight: 700;
        color: #1B4332;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F3EDE2;
        border-right: 1px solid #E2DDD5;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2D6A4F !important;
        color: #FAF7F2 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #1B4332 !important;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.25) !important;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None

if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None

# Check model availability once per run
model_status = check_model_status("models")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
status_pill_html = (
    f'<div class="status-pill status-pill-connected">🟢 AI Deep Learning Model: Connected ({model_status["filename"]})</div>'
    if model_status["connected"]
    else '<div class="status-pill status-pill-prototype">🟡 Prototype Mode: Trained AI Model Not Connected (Using Color-Space Baseline)</div>'
)

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">🥭 FruitSense AI</div>
        <div class="hero-subtitle">Smart Post-Harvest Mango Maturity & Quality Assessment Platform</div>
        {status_pill_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: System Guide & Information
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1553279768-865429fa0078?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
        use_container_width=True,
        caption="Post-Harvest Quality Assessment",
    )

    st.markdown("### 🌿 About FruitSense AI")
    st.write(
        "FruitSense AI helps farmers, distributors, and retailers evaluate "
        "mango maturity non-destructively to optimize harvest logistics, cold storage, "
        "and market timing."
    )

    st.markdown("---")
    st.markdown("### 📊 Maturity Stages")
    st.markdown(
        """
        - **🟢 Unripe:** High chlorophyll, firm, tart. Shelf life 7–14 days. Ideal for export & transit.
        - **🟡 Semi-Ripe:** Breaking stage with developing carotenoids. Shelf life 3–5 days. Best for retail display.
        - **🟠 Ripe:** Full sugar conversion, sweet aroma, soft texture. Shelf life 1–3 days. Ready for table consumption.
        """
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Integration")
    if model_status["connected"]:
        st.success(f"Connected: `{model_status['filename']}`")
    else:
        st.warning("No model weights in `models/` folder.")
        st.info(
            "To connect your deep learning model, place your trained `.onnx`, `.keras`, "
            "or `.pt` file inside the `models/` folder. The app will auto-detect it."
        )

    st.markdown("---")
    # Quick session stats
    st.markdown(f"**Session Assessments:** `{len(st.session_state.history)}`")
    if len(st.session_state.history) > 0:
        if st.button("🗑️ Clear Session History"):
            st.session_state.history = []
            st.session_state.assessment_result = None
            st.rerun()

# ---------------------------------------------------------
# Main Layout: 2 Columns (Upload/Input & Results)
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

# ---------------------------------------------------------
# Left Column: Image Selection & Preview
# ---------------------------------------------------------
with col_left:
    st.markdown('<div class="card"><div class="card-title">📤 1. Provide Mango Image</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose image source:",
        options=["Upload My Own Image", "Select a Demo Sample"],
        horizontal=True,
    )

    if input_mode == "Upload My Own Image":
        uploaded_file = st.file_uploader(
            "Upload a mango photograph (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="For best results, upload a well-lit photo of a single mango on a plain background.",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.session_state.current_image = pil_img
            st.session_state.current_image_name = uploaded_file.name

    else:
        # Demo samples selector
        sample_dir = "sample_images"
        sample_options = {
            "Sample 1: Green Unripe Mango": "sample_unripe.png",
            "Sample 2: Turning Semi-Ripe Mango": "sample_semiripe.png",
            "Sample 3: Golden Ripe Mango": "sample_ripe.png",
        }
        selected_sample_label = st.selectbox(
            "Pick a built-in demo sample:",
            options=list(sample_options.keys()),
        )
        sample_filename = sample_options[selected_sample_label]
        sample_path = os.path.join(sample_dir, sample_filename)

        if os.path.exists(sample_path):
            st.session_state.current_image = Image.open(sample_path).convert("RGB")
            st.session_state.current_image_name = sample_filename
        else:
            st.warning("Sample directory not found.")

    # Image Preview Container
    if st.session_state.current_image is not None:
        st.markdown("---")
        st.markdown("**Image Preview & Metadata:**")
        img = st.session_state.current_image
        w, h = img.size

        st.image(
            img,
            caption=f"{st.session_state.current_image_name} ({w}x{h} px)",
            use_container_width=True,
        )

        # Assess Button
        if st.button("🔍 Assess Maturity Stage", use_container_width=True):
            with st.spinner("Analyzing fruit surface & optical characteristics..."):
                result = assess_mango(img, models_dir="models")
                st.session_state.assessment_result = result

                # Append to session history
                history_entry = {
                    "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Filename": st.session_state.current_image_name,
                    "Stage": result["stage"],
                    "Assessment Method": result["assessment_method"],
                    "Confidence": (
                        result["confidence_display"]
                        if result["confidence_score"] is not None
                        else "N/A (Prototype Mode)"
                    ),
                    "Dominant Feature": result["color_breakdown"]["dominant_feature"],
                }
                st.session_state.history.append(history_entry)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Right Column: Result Panel
# ---------------------------------------------------------
with col_right:
    st.markdown('<div class="card"><div class="card-title">📋 2. Assessment Results</div>', unsafe_allow_html=True)

    result = st.session_state.assessment_result

    if result is None:
        st.info("👈 Upload an image or select a demo sample on the left, then click **Assess Maturity Stage** to view the comprehensive diagnosis.")
    else:
        stage = result["stage"]
        guidance = get_stage_guidance(stage)

        # Stage Badge Class
        badge_class = {
            "Unripe": "stage-unripe",
            "Semi-Ripe": "stage-semiripe",
            "Ripe": "stage-ripe",
        }.get(stage, "stage-semiripe")

        st.markdown(
            f"""
            <div style="text-align: center; margin: 10px 0 16px 0;">
                <div class="stage-badge {badge_class}">{guidance['icon']} {stage}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Transparency Banner (Strict Rule: Honest AI)
        if not result["is_deep_learning"]:
            st.warning(
                "⚠️ **Model Notice:** A trained deep learning model is not connected. "
                "This assessment was performed using an authentic **Computer-Vision Color Spectrum Baseline** "
                "(measuring peel chlorophyll vs. carotenoid pixel ratios). No AI confidence score was fabricated."
            )
        else:
            st.success(
                f"✅ **Deep Learning Inference:** Scored with model `{result['model_status']['filename']}`."
            )

        # Metrics Display
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Estimated Stage", stage)
        with m2:
            if result["is_deep_learning"] and result["confidence_score"] is not None:
                st.metric("Model Confidence", f"{result['confidence_score']}%")
            else:
                st.metric("Confidence Score", "N/A (Prototype)", help="Confidence scores are only displayed from actual trained model output.")

        st.markdown("---")
        st.markdown("#### 🔬 Surface Color Breakdown")
        breakdown = result["color_breakdown"]
        st.write(f"*{breakdown['dominant_feature']}*")

        # Color Spectrum Progress Bars
        st.write(f"🌿 **Green (Chlorophyll):** `{breakdown['green_pct']}%`")
        st.progress(min(1.0, breakdown["green_pct"] / 100.0))

        st.write(f"🌕 **Yellow (Carotenoid):** `{breakdown['yellow_pct']}%`")
        st.progress(min(1.0, breakdown["yellow_pct"] / 100.0))

        st.write(f"🍊 **Orange / Blush (Carotenoid):** `{breakdown['orange_pct']}%`")
        st.progress(min(1.0, breakdown["orange_pct"] / 100.0))

        # Deep Learning probabilities if available
        if result.get("class_probabilities"):
            st.markdown("---")
            st.markdown("#### 🧠 Model Softmax Probabilities")
            for cat, prob in result["class_probabilities"].items():
                st.write(f"**{cat}:** `{prob}%`")
                st.progress(min(1.0, prob / 100.0))

        st.markdown("---")
        st.markdown("#### 🚜 Post-Harvest Guidance & Action Plan")
        st.markdown(
            f"""
            <div class="insight-row">
                <span class="insight-label">⏳ Expected Shelf Life:</span> {guidance['shelf_life']}
            </div>
            <div class="insight-row">
                <span class="insight-label">🌡️ Storage Temperature:</span> {guidance['storage_temp']}
            </div>
            <div class="insight-row">
                <span class="insight-label">❄️ Chilling Injury Risk:</span> {guidance['chilling_risk']}
            </div>
            <div class="insight-row">
                <span class="insight-label">🚚 Commercial Strategy:</span> {guidance['commercial_action']}
            </div>
            <div class="insight-row">
                <span class="insight-label">🍽️ Best Use:</span> {guidance['culinary_use']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Bottom Section: Current Session Assessment History
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div class="card"><div class="card-title">📜 3. Assessment History (Current Session)</div>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info("No assessments recorded in this session yet. Run an analysis above to see history logged here.")
else:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    # Download button for CSV export
    csv_data = df_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Session Report (CSV)",
        data=csv_data,
        file_name=f"fruitsense_session_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; color: #556052; padding: 24px 0 12px 0; font-size: 0.85rem;">
        FruitSense AI • Hackathon MVP Edition • Built with Streamlit & Python •
        <em>Non-destructive fruit assessment should always be combined with agricultural best practices.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
