import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import hashlib


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor AI Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. DARK THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

h1, h2, h3 {
    color: #c9d1d9;
}

/* No Tumor */
.status-negative {
    border-left: 6px solid #2ea043;
    background: rgba(46, 160, 67, 0.10);
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
}

/* Tumor */
.status-positive {
    border-left: 6px solid #f85149;
    background: rgba(248, 81, 73, 0.10);
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
}

/* Low Risk */
.risk-low {
    border: 1px solid #2ea043;
    background: rgba(46, 160, 67, 0.15);
    color: #3fb950;
    padding: 14px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

/* Moderate Risk */
.risk-moderate {
    border: 1px solid #d29922;
    background: rgba(210, 153, 34, 0.15);
    color: #e3b341;
    padding: 14px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

/* High Risk */
.risk-high {
    border: 1px solid #f85149;
    background: rgba(248, 81, 73, 0.15);
    color: #ff7b72;
    padding: 14px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        135deg,
        #238636 0%,
        #2ea043 100%
    );
    color: white;
    font-weight: 600;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 0;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #2ea043 0%,
        #3fb950 100%
    );
    color: white;
}

/* Info card */
.info-card {
    background: rgba(56, 139, 253, 0.10);
    border: 1px solid rgba(56, 139, 253, 0.30);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}

/* Warning card */
.warning-card {
    background: rgba(210, 153, 34, 0.10);
    border: 1px solid rgba(210, 153, 34, 0.35);
    padding: 14px;
    border-radius: 8px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. CLASS MAPPING
# ============================================================

# IMPORTANT:
#
# Training output:
#
# {'glioma': 0,
#  'meningioma': 1,
#  'notumor': 2,
#  'pituitary': 3}
#
# Therefore this order MUST NOT change.

CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


# ============================================================
# 4. PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 5. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor_model.pth"
)


# ============================================================
# 6. SHA256 FUNCTION
# ============================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# 7. LOAD MODEL
# ============================================================

# IMPORTANT:
# No @st.cache_resource here temporarily.
#
# We intentionally disable caching so every Streamlit restart
# loads the actual .pth file from disk.

def load_model():

    print("\n")
    print("=" * 70)
    print("LOADING BRAIN TUMOR MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"""
Model file not found!

Expected path:

{MODEL_PATH}

Make sure brain_tumor_model.pth
is in the same folder as app.py.
"""
        )

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    model_size = (
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # SHA256
    # --------------------------------------------------------

    model_hash = get_file_hash(
        MODEL_PATH
    )

    print(
        "MODEL PATH:"
    )

    print(
        MODEL_PATH
    )

    print(
        f"\nMODEL SIZE: {model_size:.2f} MB"
    )

    print(
        "\nMODEL SHA256:"
    )

    print(
        model_hash
    )

    # --------------------------------------------------------
    # Create ResNet18
    # --------------------------------------------------------

    model = models.resnet18(
        weights=None
    )

    # --------------------------------------------------------
    # Change final layer
    # --------------------------------------------------------

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(
        num_ftrs,
        4
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    print(
        "\nCHECKPOINT TYPE:"
    )

    print(
        type(checkpoint)
    )

    # --------------------------------------------------------
    # Extract state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove module. prefix if necessary
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key.replace(
            "module.",
            ""
        )

        cleaned_state_dict[
            new_key
        ] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print(
        "\nMODEL LOADED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    return model


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/isometric/512/brain.png",
        width=90
    )

    st.title(
        "Control Panel"
    )

    st.write("---")

    st.subheader(
        "⚙️ System Status"
    )

    model = None

    try:

        model = load_model()

        st.success(
            "PyTorch Model Loaded Successfully"
        )

    except Exception as e:

        st.error(
            f"Error loading model:\n{e}"
        )

    st.write("---")

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Information"
    )

    st.write(
        "**Architecture:** ResNet18"
    )

    st.write(
        "**Classes:** 4"
    )

    st.write(
        "**Input:** 224 × 224"
    )

    st.write(
        "**Device:** CPU"
    )

    st.write("---")

    # --------------------------------------------------------
    # MODEL FILE DEBUG
    # --------------------------------------------------------

    st.subheader(
        "📦 Loaded Model File"
    )

    st.code(
        MODEL_PATH,
        language="text"
    )

    if os.path.exists(
        MODEL_PATH
    ):

        model_size = (
            os.path.getsize(
                MODEL_PATH
            ) / (1024 * 1024)
        )

        model_hash = get_file_hash(
            MODEL_PATH
        )

        st.success(
            f"Model found\n\n"
            f"Size: {model_size:.2f} MB"
        )

        st.write(
            "**SHA256:**"
        )

        st.code(
            model_hash,
            language="text"
        )

    else:

        st.error(
            "brain_tumor_model.pth was not found!"
        )

    st.write("---")

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    st.subheader(
        "📋 Preprocessing"
    )

    st.info(
        "The preprocessing below exactly matches "
        "the preprocessing used during training."
    )

    st.write(
        "Resize → 224 × 224"
    )

    st.write(
        "ToTensor → Enabled"
    )

    st.write(
        "ImageNet Normalization → Enabled"
    )

    st.write("---")

    # --------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------

    st.subheader(
        "🏷️ Class Mapping"
    )

    st.write(
        "0 → Glioma Tumor"
    )

    st.write(
        "1 → Meningioma Tumor"
    )

    st.write(
        "2 → No Tumor"
    )

    st.write(
        "3 → Pituitary Tumor"
    )


# ============================================================
# 9. MAIN TITLE
# ============================================================

st.title(
    "🧠 Brain Tumor AI Classification & Analysis"
)

st.caption(
    "Deep Learning Decision Support System for MRI Scans"
)

st.write("---")


# ============================================================
# 10. MAIN COLUMNS
# ============================================================

col1, col2 = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# 11. UPLOAD MRI
# ============================================================

with col1:

    st.subheader(
        "📤 Upload MRI Scan"
    )

    uploaded_file = st.file_uploader(
        "Drop your Brain MRI image here...",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption=(
                    f"Uploaded MRI: "
                    f"{uploaded_file.name}"
                ),
                use_container_width=True
            )

            # Show image information
            st.write(
                f"**Filename:** {uploaded_file.name}"
            )

            st.write(
                f"**Original Size:** "
                f"{image.size[0]} × {image.size[1]}"
            )

        except Exception as e:

            st.error(
                f"Could not open image: {e}"
            )

            image = None

    else:

        image = None

        st.info(
            "Please upload an MRI image."
        )


# ============================================================
# 12. DIAGNOSTIC ANALYSIS
# ============================================================

with col2:

    st.subheader(
        "📊 Diagnostic Analysis"
    )

    if image is not None:

        analyze_button = st.button(
            "🚀 Run AI Diagnostic Analysis"
        )

        if analyze_button:

            if model is None:

                st.error(
                    "Model is not loaded correctly."
                )

            else:

                with st.spinner(
                    "Processing MRI scan..."
                ):

                    # ====================================================
                    # PREPROCESSING
                    # ====================================================

                    transform = transforms.Compose([

                        transforms.Resize(
                            (224, 224)
                        ),

                        transforms.ToTensor(),

                        transforms.Normalize(
                            mean=[
                                0.485,
                                0.456,
                                0.406
                            ],
                            std=[
                                0.229,
                                0.224,
                                0.225
                            ]
                        )

                    ])


                    # ====================================================
                    # IMAGE → TENSOR
                    # ====================================================

                    input_tensor = transform(
                        image
                    ).unsqueeze(0)


                    # ====================================================
                    # DEBUG INPUT
                    # ====================================================

                    print("\n")
                    print("=" * 70)
                    print(
                        "STREAMLIT IMAGE DEBUG"
                    )
                    print("=" * 70)

                    print(
                        "Filename:",
                        uploaded_file.name
                    )

                    print(
                        "Original image size:",
                        image.size
                    )

                    print(
                        "Tensor shape:",
                        input_tensor.shape
                    )

                    print(
                        "Tensor dtype:",
                        input_tensor.dtype
                    )

                    print(
                        "Tensor min:",
                        input_tensor.min().item()
                    )

                    print(
                        "Tensor max:",
                        input_tensor.max().item()
                    )


                    # ====================================================
                    # MODEL INFERENCE
                    # ====================================================

                    with torch.no_grad():

                        output = model(
                            input_tensor
                        )

                        probabilities = torch.softmax(
                            output,
                            dim=1
                        )[0]


                    # ====================================================
                    # DEBUG PROBABILITIES
                    # ====================================================

                    print(
                        "\nSTREAMLIT PREDICTION DEBUG:"
                    )

                    print(
                        "-" * 50
                    )

                    for i, class_name in enumerate(
                        CLASS_NAMES
                    ):

                        probability = (
                            probabilities[i].item()
                            * 100
                        )

                        print(
                            f"{i} - "
                            f"{class_name}: "
                            f"{probability:.6f}%"
                        )

                    print(
                        "-" * 50
                    )


                    # ====================================================
                    # PREDICTED CLASS
                    # ====================================================

                    predicted_index = torch.argmax(
                        probabilities
                    ).item()

                    predicted_class = CLASS_NAMES[
                        predicted_index
                    ]

                    confidence = (
                        probabilities[
                            predicted_index
                        ].item()
                        * 100
                    )


                    # ====================================================
                    # INDIVIDUAL PROBABILITIES
                    # ====================================================

                    glioma_prob = (
                        probabilities[0].item()
                        * 100
                    )

                    meningioma_prob = (
                        probabilities[1].item()
                        * 100
                    )

                    no_tumor_prob = (
                        probabilities[2].item()
                        * 100
                    )

                    pituitary_prob = (
                        probabilities[3].item()
                        * 100
                    )


                    # ====================================================
                    # TUMOR PROBABILITY
                    # ====================================================

                    tumor_probability = (
                        100.0
                        - no_tumor_prob
                    )


                    # ====================================================
                    # RESULT
                    # ====================================================

                    if predicted_class == "No Tumor":

                        st.markdown(
                            f"""
                            <div class="status-negative">

                                <h3 style="color:#3fb950;">
                                    ✅ Result: No Tumor Detected
                                </h3>

                                <p>
                                    <b>Model Confidence:</b>
                                    {confidence:.2f}%
                                </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="status-positive">

                                <h3 style="color:#ff7b72;">
                                    ⚠️ Result: {predicted_class}
                                </h3>

                                <p>
                                    <b>Model Confidence:</b>
                                    {confidence:.2f}%
                                </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # ====================================================
                    # TUMOR PROBABILITY
                    # ====================================================

                    st.write("---")

                    st.subheader(
                        "🧠 Tumor Probability Assessment"
                    )

                    metric1, metric2 = st.columns(2)

                    with metric1:

                        st.metric(
                            "Tumor Probability",
                            f"{tumor_probability:.2f}%"
                        )

                    with metric2:

                        st.metric(
                            "No Tumor Probability",
                            f"{no_tumor_prob:.2f}%"
                        )


                    # ====================================================
                    # RISK STAGE
                    # ====================================================

                    if tumor_probability < 20:

                        st.markdown(
                            """
                            <div class="risk-low">

                                🟢 LOW AI-INDICATED RISK

                                <br><br>

                                The model assigns a relatively
                                low probability to tumor classes.

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    elif tumor_probability < 50:

                        st.markdown(
                            """
                            <div class="risk-moderate">

                                🟡 MODERATE AI-INDICATED RISK

                                <br><br>

                                The model assigns a moderate
                                probability to tumor classes.

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="risk-high">

                                🔴 HIGH AI-INDICATED RISK

                                <br><br>

                                The model assigns a high
                                probability to tumor classes.

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # ====================================================
                    # PROGRESS BAR
                    # ====================================================

                    st.write("")

                    st.progress(
                        min(
                            int(tumor_probability),
                            100
                        )
                    )


                    # ====================================================
                    # PROBABILITY TABLE
                    # ====================================================

                    st.write("---")

                    st.subheader(
                        "📈 Prediction Probabilities"
                    )

                    probability_data = {

                        "Class": [
                            "Glioma Tumor",
                            "Meningioma Tumor",
                            "No Tumor",
                            "Pituitary Tumor"
                        ],

                        "Probability (%)": [

                            glioma_prob,

                            meningioma_prob,

                            no_tumor_prob,

                            pituitary_prob

                        ]

                    }


                    df_probs = pd.DataFrame(
                        probability_data
                    )


                    st.dataframe(
                        df_probs,
                        hide_index=True,
                        use_container_width=True
                    )


                    # ====================================================
                    # BAR CHART
                    # ====================================================

                    chart_df = df_probs.set_index(
                        "Class"
                    )

                    st.bar_chart(
                        chart_df
                    )


                    # ====================================================
                    # AI INTERPRETATION
                    # ====================================================

                    st.write("---")

                    st.subheader(
                        "💡 AI Interpretation"
                    )

                    if predicted_class == "No Tumor":

                        st.info(
                            f"""
The model classified this image as
**No Tumor** with a model confidence
of **{confidence:.2f}%**.

Combined probability assigned to the
three tumor classes:

**{tumor_probability:.2f}%**
"""
                        )

                    else:

                        st.warning(
                            f"""
The model classified this image as
**{predicted_class}** with a model confidence
of **{confidence:.2f}%**.

Combined probability assigned to
tumor classes:

**{tumor_probability:.2f}%**
"""
                        )


                    # ====================================================
                    # MEDICAL DISCLAIMER
                    # ====================================================

                    st.markdown(
                        """
                        <div class="warning-card">

                        ⚠️ <b>Important:</b>

                        This application is an AI-assisted
                        educational/research tool.

                        The displayed classification and
                        probabilities are model outputs and
                        are not a medical diagnosis.

                        MRI interpretation should be performed
                        by a qualified medical professional.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    else:

        st.info(
            "Upload an MRI image and click "
            "'Run AI Diagnostic Analysis'."
        )