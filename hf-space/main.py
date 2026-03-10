import streamlit as st
import tensorflow as tf
import numpy as np
import os
import re
import base64 as _b64
import requests as _req_lib
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

MODEL_PATH = "trained_plant_disease_model.keras"
TEST_DIR = "test"
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.50  # minimum confidence to accept a prediction as valid crop image

_GEMINI_API_KEY = "AIzaSyBGLJQP6gVfbcnTkmXYoxHPTOhXqQT_O8I"
_GEMINI_MODEL = "gemma-3-27b-it"
_GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_API_KEY}"

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Mapping from test image filename prefixes to class names
FILENAME_TO_CLASS = {
    'AppleCedarRust':       'Apple___Cedar_apple_rust',
    'AppleScab':            'Apple___Apple_scab',
    'CornCommonRust':       'Corn_(maize)___Common_rust_',
    'PotatoEarlyBlight':    'Potato___Early_blight',
    'PotatoHealthy':        'Potato___healthy',
    'TomatoEarlyBlight':    'Tomato___Early_blight',
    'TomatoHealthy':        'Tomato___healthy',
    'TomatoYellowCurlVirus':'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
}

# ── Disease Treatment Database ──
# Maps each disease class to cause, treatment, dosage, and application method
DISEASE_TREATMENT_DB = {
    'Apple___Apple_scab': {
        'cause': 'Fungal infection caused by Venturia inaequalis; spreads in cool, wet weather.',
        'treatment': 'Captan or Myclobutanil fungicide',
        'dosage': '2–3 g/L of water',
        'application': 'Spray every 7–10 days during wet weather. Remove and destroy fallen infected leaves.',
    },
    'Apple___Black_rot': {
        'cause': 'Fungus Botryosphaeria obtusa; infects through wounds and dead wood.',
        'treatment': 'Captan or Thiophanate-methyl fungicide',
        'dosage': '2 g/L of water',
        'application': 'Apply at petal fall and repeat every 10–14 days. Prune out mummified fruits and cankers.',
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Fungus Gymnosporangium juniperi-virginianae; requires cedar/juniper as alternate host.',
        'treatment': 'Myclobutanil or Mancozeb fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Spray at pink bud stage and repeat every 7–10 days through petal fall. Remove nearby junipers if possible.',
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'cause': 'Fungus Podosphaera clandestina; thrives in warm, dry days with cool nights.',
        'treatment': 'Sulfur-based fungicide or Myclobutanil',
        'dosage': '3 g/L of water (sulfur) or 1 mL/L (Myclobutanil)',
        'application': 'Spray when first symptoms appear, repeat every 10–14 days. Ensure good air circulation by pruning.',
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'cause': 'Fungus Cercospora zeae-maydis; favored by prolonged leaf wetness and humidity.',
        'treatment': 'Azoxystrobin (Quadris) or Pyraclostrobin fungicide',
        'dosage': '1 mL/L of water',
        'application': 'Apply at early tassel stage. Rotate crops and use resistant hybrids for prevention.',
    },
    'Corn_(maize)___Common_rust_': {
        'cause': 'Fungus Puccinia sorghi; spreads via wind-borne spores in moderate temperatures.',
        'treatment': 'Mancozeb or Propiconazole fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Spray at first sign of pustules, repeat every 7–10 days. Plant resistant varieties.',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cause': 'Fungus Exserohilum turcicum; thrives in moderate temps with heavy dew.',
        'treatment': 'Azoxystrobin or Propiconazole fungicide',
        'dosage': '1 mL/L of water',
        'application': 'Apply at first sign of lesions. Use crop rotation and resistant hybrids.',
    },
    'Grape___Black_rot': {
        'cause': 'Fungus Guignardia bidwellii; spreads in warm, humid conditions.',
        'treatment': 'Myclobutanil or Mancozeb fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Spray from shoot emergence through veraison at 7–10 day intervals. Remove mummified berries.',
    },
    'Grape___Esca_(Black_Measles)': {
        'cause': 'Complex of fungi (Phaeomoniella, Phaeoacremonium); wood-inhabiting pathogens.',
        'treatment': 'No reliable chemical cure; apply Trichoderma biocontrol agent',
        'dosage': '5 g/L of water (Trichoderma formulation)',
        'application': 'Apply to pruning wounds immediately after cutting. Remove heavily infected vines.',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'cause': 'Fungus Pseudocercospora vitis; spreads in warm, wet conditions.',
        'treatment': 'Copper-based fungicide (Bordeaux mixture) or Mancozeb',
        'dosage': '2–3 g/L of water',
        'application': 'Spray every 10–14 days during growing season. Improve canopy air circulation.',
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'cause': 'Bacterium Candidatus Liberibacter; spread by Asian citrus psyllid insect.',
        'treatment': 'No cure exists. Control psyllid vector with Imidacloprid insecticide',
        'dosage': '0.5 mL/L of water (soil drench) or 0.3 mL/L (foliar)',
        'application': 'Apply soil drench in early spring. Foliar spray every 3–4 weeks. Remove severely infected trees.',
    },
    'Peach___Bacterial_spot': {
        'cause': 'Bacterium Xanthomonas arboricola pv. pruni; favored by warm, wet weather.',
        'treatment': 'Copper hydroxide bactericide + Oxytetracycline',
        'dosage': '2 g/L copper hydroxide',
        'application': 'Apply copper sprays at leaf fall and again at bud swell. Avoid overhead irrigation.',
    },
    'Pepper,_bell___Bacterial_spot': {
        'cause': 'Bacterium Xanthomonas campestris pv. vesicatoria; spreads via contaminated seed and rain splash.',
        'treatment': 'Copper-based bactericide + Mancozeb tank mix',
        'dosage': '2 g copper + 2.5 g Mancozeb per liter of water',
        'application': 'Spray every 7–10 days during wet weather. Use disease-free seed.',
    },
    'Potato___Early_blight': {
        'cause': 'Fungus Alternaria solani; attacks stressed or aging foliage.',
        'treatment': 'Chlorothalonil or Mancozeb fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Begin spraying when lower leaves show spots. Repeat every 7–10 days.',
    },
    'Potato___Late_blight': {
        'cause': 'Oomycete Phytophthora infestans; thrives in cool, moist weather.',
        'treatment': 'Metalaxyl + Mancozeb (Ridomil Gold MZ)',
        'dosage': '2.5 g/L of water',
        'application': 'Apply preventively every 7 days in conducive weather. Destroy infected tubers.',
    },
    'Squash___Powdery_mildew': {
        'cause': 'Fungus Podosphaera xanthii; favored by warm days, cool nights, and shade.',
        'treatment': 'Sulfur fungicide or Potassium bicarbonate',
        'dosage': '3 g/L (sulfur) or 5 g/L (potassium bicarbonate)',
        'application': 'Spray at first sign of white patches. Repeat every 7–10 days. Ensure good spacing.',
    },
    'Strawberry___Leaf_scorch': {
        'cause': 'Fungus Diplocarpon earlianum; spreads via splashing rain on infected leaf debris.',
        'treatment': 'Captan or Myclobutanil fungicide',
        'dosage': '2 g/L of water',
        'application': 'Spray after bloom and repeat every 10–14 days. Remove debris after harvest.',
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Bacterium Xanthomonas vesicatoria; spreads through seed, rain splash, and tools.',
        'treatment': 'Copper hydroxide + Mancozeb fungicide',
        'dosage': '2 g copper + 2.5 g Mancozeb per liter of water',
        'application': 'Apply every 5–7 days during wet conditions. Use disease-free transplants.',
    },
    'Tomato___Early_blight': {
        'cause': 'Fungus Alternaria solani; attacks lower/older leaves first.',
        'treatment': 'Chlorothalonil or Mancozeb fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Begin when first lesions appear. Spray every 7–10 days. Mulch to prevent soil splash.',
    },
    'Tomato___Late_blight': {
        'cause': 'Oomycete Phytophthora infestans; spreads rapidly in cool, wet conditions.',
        'treatment': 'Mancozeb or Chlorothalonil fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Apply every 7 days. Remove and destroy infected leaves immediately.',
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Fungus Passalora fulva (Cladosporium fulvum); thrives in high humidity greenhouses.',
        'treatment': 'Chlorothalonil or Copper-based fungicide',
        'dosage': '2 g/L of water',
        'application': 'Improve ventilation and reduce humidity. Spray every 7–10 days. Avoid overhead watering.',
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Fungus Septoria lycopersici; spreads via rain splash from infected crop debris.',
        'treatment': 'Chlorothalonil or Mancozeb fungicide',
        'dosage': '2.5 g/L of water',
        'application': 'Spray when first spots appear. Repeat every 7–10 days. Remove lower infected leaves.',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Pest infestation by Tetranychus urticae; thrives in hot, dry conditions.',
        'treatment': 'Abamectin or Neem oil (organic option)',
        'dosage': '0.5 mL/L (Abamectin) or 5 mL/L (Neem oil)',
        'application': 'Spray undersides of leaves thoroughly. Repeat every 5–7 days. Use predatory mites for biocontrol.',
    },
    'Tomato___Target_Spot': {
        'cause': 'Fungus Corynespora cassiicola; favored by warm, humid conditions.',
        'treatment': 'Chlorothalonil or Azoxystrobin fungicide',
        'dosage': '2 g/L of water',
        'application': 'Spray at first sign of concentric ring spots. Repeat every 7–10 days.',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Viral (TYLCV); transmitted by whiteflies (Bemisia tabaci).',
        'treatment': 'No cure for virus. Control whitefly vector with Imidacloprid',
        'dosage': '0.5 mL/L of water',
        'application': 'Apply soil drench at transplanting. Use reflective mulch and yellow sticky traps. Remove infected plants.',
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Viral (ToMV); mechanically transmitted via contaminated hands, tools, and seed.',
        'treatment': 'No chemical cure. Remove infected plants immediately',
        'dosage': 'N/A — prevention only',
        'application': 'Sanitize tools with 10% bleach solution. Wash hands before handling plants. Use resistant varieties.',
    },
}


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained Keras model (cached so it only loads once)."""
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, image_path_or_file):
    """
    Run prediction on a single image.
    Returns (predicted_class_index, confidence, probabilities_array).
    """
    image = tf.keras.preprocessing.image.load_img(image_path_or_file, target_size=IMAGE_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # batch dimension
    predictions = model.predict(input_arr, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    return predicted_index, confidence, predictions[0]


def validate_is_leaf(uploaded_file):
    """
    Use Gemini vision to check if the uploaded image is actually a plant/crop leaf.
    Returns True if it appears to be a leaf, False otherwise.
    """
    try:
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        img_b64 = _b64.b64encode(img_bytes).decode("utf-8")

        mime = getattr(uploaded_file, 'type', 'image/jpeg') or 'image/jpeg'

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": (
                        "Look at this image carefully. Does it contain a real plant leaf or crop leaf? "
                        "Answer ONLY with the single word YES or NO. "
                        "Answer YES only if the image clearly shows a real plant, crop, or leaf. "
                        "Answer NO for screenshots, documents, text, animals, people, objects, "
                        "error pages, or anything that is not a plant leaf."
                    )},
                    {"inlineData": {"mimeType": mime, "data": img_b64}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10,
            },
        }

        resp = _req_lib.post(
            _GEMINI_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            answer = (
                resp.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
                .upper()
            )
            return answer.startswith("YES")
        # If API fails, allow the image through (fail-open)
        return True
    except Exception:
        return True  # fail-open: don't block if validation fails


def parse_class_name(class_name):
    """
    Parse a class name like 'Apple___Apple_scab' into a dict:
      { 'plant': 'Apple', 'is_healthy': False, 'disease': 'Apple Scab' }
    """
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').replace(',', ', ').strip()
    label = parts[1] if len(parts) > 1 else ''

    is_healthy = label.lower() == 'healthy'
    if is_healthy:
        disease = None
    else:
        # Clean up the disease name for display
        disease = label.replace('_', ' ').strip()
        # Capitalize first letter of each word
        disease = disease.title()

    return {
        'plant': plant,
        'is_healthy': is_healthy,
        'disease': disease,
    }


def filename_to_ground_truth(filename):
    """
    Extract ground-truth class name from a test image filename.
    e.g. 'AppleCedarRust1.JPG' -> 'Apple___Cedar_apple_rust'
    Returns the class name string or None if unrecognized.
    """
    name_no_ext = os.path.splitext(filename)[0]
    # Strip trailing digits
    prefix = re.sub(r'\d+$', '', name_no_ext)
    return FILENAME_TO_CLASS.get(prefix, None)


def get_test_images():
    """Return list of (filename, full_path) for loose .JPG/.jpg/.png images in TEST_DIR."""
    if not os.path.isdir(TEST_DIR):
        return []
    images = []
    for f in sorted(os.listdir(TEST_DIR)):
        full = os.path.join(TEST_DIR, f)
        if os.path.isfile(full) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append((f, full))
    return images


def format_result(pred_index, confidence):
    """Return a formatted dict from a prediction index."""
    class_name = CLASS_NAMES[pred_index]
    info = parse_class_name(class_name)
    info['confidence'] = confidence
    info['raw_class'] = class_name
    return info


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AgriSens – Plant Disease Detection", layout="wide")

# Sidebar
st.sidebar.title("🌿 AgriGuard AI")
app_mode = st.sidebar.selectbox(
    "Select Page",
    ["HOME", "DISEASE RECOGNITION", "FERTILIZER RECOMMENDATION", "AGRI BOT ADVISOR"]
)

# Banner image
if os.path.exists("Diseases.png"):
    banner = Image.open("Diseases.png")
    st.image(banner, use_container_width=True)


# ─────────── HOME ───────────
if app_mode == "HOME":

    # ── Custom CSS for the entire home page ──
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #1a5632 0%, #2d8659 40%, #3a7d44 70%, #1a5632 100%);
            padding: 60px 40px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.05)"/><circle cx="80" cy="40" r="3" fill="rgba(255,255,255,0.04)"/><circle cx="50" cy="80" r="1.5" fill="rgba(255,255,255,0.06)"/></svg>');
            pointer-events: none;
        }
        .hero-section .welcome-tag {
            color: #d4a843;
            font-size: 0.95rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .hero-section h1 {
            color: #ffffff;
            font-size: 3rem;
            font-weight: 700;
            margin: 10px 0 15px 0;
            line-height: 1.2;
        }
        .hero-section h1 span {
            color: #d4a843;
        }
        .hero-section .subtitle {
            color: rgba(255,255,255,0.85);
            font-size: 1.15rem;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .feature-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e8f0e8;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 12px;
        }
        .feature-card h3 {
            color: #1a5632;
            font-size: 1.15rem;
            margin: 8px 0;
        }
        .feature-card p {
            color: #555;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .section-header {
            text-align: center;
            margin: 40px 0 25px 0;
        }
        .section-header h2 {
            color: #1a5632;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .section-header .accent-line {
            width: 60px;
            height: 3px;
            background: #d4a843;
            margin: 10px auto;
            border-radius: 2px;
        }
        .step-card {
            background: linear-gradient(135deg, #f0f7f0, #e8f5e8);
            border-radius: 12px;
            padding: 25px 20px;
            text-align: center;
            border-left: 4px solid #2d8659;
        }
        .step-number {
            background: #1a5632;
            color: #fff;
            width: 36px; height: 36px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 10px;
        }
        .crop-badge {
            display: inline-block;
            background: linear-gradient(135deg, #e8f5e8, #d4edda);
            color: #1a5632;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9rem;
            font-weight: 500;
            border: 1px solid #c3e6cb;
        }
        .cta-section {
            background: linear-gradient(135deg, #d4a843 0%, #c9952e 100%);
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            margin-top: 30px;
        }
        .cta-section h2 { color: #fff; margin-bottom: 10px; }
        .cta-section p { color: rgba(255,255,255,0.9); font-size: 1.05rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ──
    st.markdown("""
    <h1 style="text-align:center; color:#1a5632; font-size:2.2rem; font-weight:700; margin-bottom:30px;">
        🌿 AgriGuard AI – Smart Crop Disease Detection &amp; Farmer Advisory
    </h1>
    """, unsafe_allow_html=True)

    # ── Feature Cards ──
    st.markdown("""
    <div class="section-header">
        <h2>Why AgriSens?</h2>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>AI-Powered Detection</h3>
            <p>Deep learning CNN model trained on 70,000+ crop leaf images for highly accurate disease identification.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Instant Results</h3>
            <p>Get disease diagnosis in seconds. Simply upload a photo of the leaf and our AI does the rest.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3>38+ Disease Classes</h3>
            <p>Identifies diseases across 14 crop types including scab, blight, rust, mold, mosaic virus, and more.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div class="section-header">
        <h2>How It Works</h2>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <h4>📸 Upload Image</h4>
            <p>Take a clear photo of the plant leaf and upload it on the Disease Recognition page.</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <h4>🧠 AI Analysis</h4>
            <p>Our trained CNN model analyzes the leaf pattern, color, and texture to detect abnormalities.</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <h4>📋 Get Results</h4>
            <p>Receive instant diagnosis: Healthy or Diseased, disease name, and confidence score.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Supported Crops ──
    st.markdown("""
    <div class="section-header">
        <h2>Supported Crops</h2>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)

    crops = [
        "🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn",
        "🍇 Grape", "🍊 Orange", "🍑 Peach", "🫑 Pepper",
        "🥔 Potato", "🫐 Raspberry", "🫘 Soybean", "🎃 Squash",
        "🍓 Strawberry", "🍅 Tomato"
    ]
    badges_html = "".join(f'<span class="crop-badge">{c}</span>' for c in crops)
    st.markdown(f'<div style="text-align:center;">{badges_html}</div>', unsafe_allow_html=True)

    # ── Call to Action ──
    st.markdown("""
    <div class="cta-section">
        <h2>🌿 Ready to Protect Your Crops?</h2>
        <p>Select <b>"DISEASE RECOGNITION"</b> from the sidebar to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────── DISEASE RECOGNITION ───────────
elif app_mode == "DISEASE RECOGNITION":
    st.header("🔬 Disease Recognition")
    st.write("Upload a plant leaf image to check if it is healthy or diseased.")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(test_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🔍 Analyze", use_container_width=True):
                with st.spinner("Verifying image…"):
                    is_leaf = validate_is_leaf(test_image)

                st.markdown("---")

                if not is_leaf:
                    # Image is NOT a plant leaf
                    st.subheader("🚜 Farmer Advisor")
                    st.warning(
                        "👨‍🌾 **Hello! I am your Farmer Advisor.**\n\n"
                        "I specialize in identifying **crop leaf diseases**. "
                        "The image you uploaded **does not appear to contain a plant leaf**.\n\n"
                        "🌿 **Please upload a clear picture of a crop/plant leaf** "
                        "so I can analyze it for diseases."
                    )
                    st.info(
                        "💡 **Tips for best results:**\n"
                        "- Upload a close-up photo of a single plant leaf\n"
                        "- Make sure the leaf is clearly visible and well-lit\n"
                        "- Avoid screenshots, documents, or non-plant images\n"
                        "- Supported crops: Apple, Blueberry, Cherry, Corn, Grape, "
                        "Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, "
                        "Strawberry, and Tomato"
                    )
                else:
                    with st.spinner("Analyzing disease…"):
                        test_image.seek(0)
                        model = load_model()
                        pred_index, confidence, _ = predict_image(model, test_image)
                        result = format_result(pred_index, confidence)

                    # Check if the image is a valid crop image
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.subheader("🚜 Farmer Advisor")
                        st.warning(
                            "👨‍🌾 **Hello! I am your Farmer Advisor.**\n\n"
                            "I specialize in identifying **crop leaf diseases**. "
                            "The image you uploaded doesn't appear to be a recognizable crop leaf image.\n\n"
                            "🌿 **Please upload a clear picture of a crop/plant leaf** "
                            "so I can analyze it for diseases."
                        )
                        st.info(
                            "💡 **Tips for best results:**\n"
                            "- Upload a close-up photo of a single plant leaf\n"
                            "- Make sure the leaf is clearly visible and well-lit\n"
                            "- Supported crops: Apple, Blueberry, Cherry, Corn, Grape, "
                            "Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, "
                            "Strawberry, and Tomato"
                        )
                    else:
                        conf_pct = result['confidence'] * 100

                        if result['is_healthy']:
                            # ── Healthy Plant Card ──
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-left: 5px solid #2e7d32;
                                        border-radius: 12px; padding: 28px 32px; margin: 10px 0;">
                                <h2 style="color: #2e7d32; margin: 0 0 6px 0;">✅ {result['plant']} — Healthy</h2>
                                <p style="color: #555; font-size: 1.05rem; margin: 0;">Confidence: <b>{conf_pct:.1f}%</b></p>
                                <hr style="border: none; border-top: 1px solid #a5d6a7; margin: 14px 0;">
                                <p style="color: #2e7d32; font-size: 1rem; margin: 0;">
                                    🎉 Great news! Your <b>{result['plant']}</b> plant looks healthy.
                                    Keep up the good agricultural practices!
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # ── Diseased Plant Treatment Card ──
                            treatment = DISEASE_TREATMENT_DB.get(result['raw_class'], None)

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #fff8e1, #fff3e0); border-left: 5px solid #e65100;
                                        border-radius: 12px; padding: 28px 32px; margin: 10px 0;
                                        box-shadow: 0 2px 12px rgba(0,0,0,0.07);">
                                <h2 style="color: #bf360c; margin: 0 0 4px 0;">🦠 {result['plant']} — {result['disease']}</h2>
                                <p style="color: #6d4c41; font-size: 1.05rem; margin: 0 0 14px 0;">Confidence: <b>{conf_pct:.1f}%</b></p>
                                <hr style="border: none; border-top: 1px solid #ffe0b2; margin: 0 0 16px 0;">
                            """, unsafe_allow_html=True)

                            if treatment:
                                st.markdown(f"""
                                <div style="padding: 0 32px 6px 32px;">
                                    <div style="margin-bottom: 16px;">
                                        <span style="font-size: 1.1rem; font-weight: 600; color: #4e342e;">🔍 Cause</span>
                                        <p style="color: #555; margin: 4px 0 0 0;">{treatment['cause']}</p>
                                    </div>
                                    <div style="margin-bottom: 16px;">
                                        <span style="font-size: 1.1rem; font-weight: 600; color: #2e7d32;">💊 Recommended Treatment</span>
                                        <p style="color: #555; margin: 4px 0 0 0;">• {treatment['treatment']}</p>
                                    </div>
                                    <div style="margin-bottom: 16px;">
                                        <span style="font-size: 1.1rem; font-weight: 600; color: #1565c0;">📏 Dosage</span>
                                        <p style="color: #555; margin: 4px 0 0 0;">• {treatment['dosage']}</p>
                                    </div>
                                    <div style="margin-bottom: 6px;">
                                        <span style="font-size: 1.1rem; font-weight: 600; color: #6a1b9a;">🌿 Application Method</span>
                                        <p style="color: #555; margin: 4px 0 0 0;">• {treatment['application']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="padding: 0 32px 6px 32px;">
                                    <p style="color: #888;"><i>Treatment information not available in the local database.
                                    See the AI-generated advice below for recommendations.</i></p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

                        # Raw class (for debugging/reference)
                        with st.expander("🔧 Technical Details"):
                            st.write(f"Predicted class: `{result['raw_class']}`")
                            st.write(f"Class index: `{pred_index}`")

                        # ── AgriBot Treatment Advice ──
                        if result['is_healthy']:
                            advice_prompt = (
                                f"The plant is a **{result['plant']}** and it appears **healthy**. "
                                f"Provide brief practical tips to keep this plant healthy, prevent common diseases, "
                                f"and boost yield. Use bullet points. Keep it concise and farmer-friendly."
                            )
                        else:
                            advice_prompt = (
                                f"A **{result['plant']}** plant has been diagnosed with **{result['disease']}** "
                                f"(confidence: {conf_pct:.1f}%). As an expert agronomist, provide:\n"
                                f"1. **About the Disease** — What it is and what causes it (1-2 lines)\n"
                                f"2. **Symptoms** — Key visual signs on the leaf/plant\n"
                                f"3. **Organic Treatment** — Safe, natural remedies to cure it without damaging the crop\n"
                                f"4. **Chemical Treatment** — Recommended fungicides/pesticides (if necessary)\n"
                                f"5. **Prevention Tips** — How to avoid this disease in the future\n\n"
                                f"Keep your answer concise, structured with bullet points, and farmer-friendly. "
                                f"Always prioritize methods that DON'T damage the crop."
                            )

                        _sys = (
                            "You are AgriBot, an expert AI Agronomist. "
                            "Give practical, actionable farming advice. Use markdown formatting."
                        )

                        st.markdown("---")
                        st.subheader("🌱 AgriBot Treatment Advice")

                        with st.spinner("🤖 AgriBot is preparing advice..."):
                            try:
                                _payload = {
                                    "contents": [{"role": "user", "parts": [
                                        {"text": f"[System: {_sys}]\n\n{advice_prompt}"}
                                    ]}],
                                    "generationConfig": {
                                        "temperature": 0.7,
                                        "topK": 40,
                                        "topP": 0.95,
                                        "maxOutputTokens": 1024,
                                    },
                                }
                                _resp = _req_lib.post(
                                    _GEMINI_API_URL,
                                    json=_payload,
                                    headers={"Content-Type": "application/json"},
                                    timeout=30,
                                )
                                if _resp.status_code == 200:
                                    _data = _resp.json()
                                    _advice = (
                                        _data.get("candidates", [{}])[0]
                                        .get("content", {})
                                        .get("parts", [{}])[0]
                                        .get("text", "Could not generate advice.")
                                    )
                                else:
                                    _advice = f"⚠️ Could not fetch advice (API error {_resp.status_code})."
                            except Exception as _e:
                                _advice = f"⚠️ Connection error: {str(_e)}"

                        st.markdown(_advice)


# ─────────── AGRI BOT ADVISOR ───────────
elif app_mode == "AGRI BOT ADVISOR":

    AGRIBOT_SYSTEM_PROMPT = """
You are AgriBot, an expert AI Agronomist and Farmer Advisory Chatbot.
Your primary goal is to help farmers identify crop diseases early, optimize irrigation, and provide fertilizer recommendations.
You are knowledgeable, practical, encouraging, and deeply understand agriculture, soil science, and botany.
Keep your answers concise, structured (use bullet points or bold text for readability), and actionable.
Always prioritize sustainable and accessible farming practices.
If a user uploads an image, analyze it for signs of crop disease, nutrient deficiency, or pests, and provide a clear diagnosis and treatment plan.
"""


    # ── Header ──
    st.markdown("""
    <div style="display:flex; align-items:center; gap:14px; padding:12px 0 20px 0; border-bottom:1px solid #e0e0e0; margin-bottom:20px;">
        <div style="width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg,#1a5632,#2d8659);
                    display:flex; align-items:center; justify-content:center; position:relative;">
            <span style="font-size:1.5rem;">🌿</span>
            <div style="width:12px; height:12px; background:#22c55e; border-radius:50%; border:2px solid white;
                        position:absolute; bottom:0; right:0;"></div>
        </div>
        <div>
            <h2 style="margin:0; font-size:1.4rem; color:#1a5632;">AgriBot Advisor</h2>
            <span style="font-size:0.85rem; color:#888;">Powered by Gemma 3 · Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ──
    if "agribot_messages" not in st.session_state:
        st.session_state.agribot_messages = [
            {
                "role": "assistant",
                "content": (
                    "**Hello! I am your AI Agronomist.** 🌱\n\n"
                    "I can help you:\n"
                    "- 🔬 **Identify crop diseases** from leaf photos\n"
                    "- 💧 **Optimize irrigation** schedules\n"
                    "- 🧪 **Recommend fertilizers** for your soil\n\n"
                    "Upload a photo of your crop or describe your issue below!"
                ),
            }
        ]

    # ── Display chat history ──
    for msg in st.session_state.agribot_messages:
        avatar = "🌿" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("image"):
                st.image(msg["image"], width=250)
            st.markdown(msg["content"])

    # ── Image upload ──
    uploaded_img = st.file_uploader(
        "📷 Upload a crop/leaf image for analysis (optional)",
        type=["jpg", "jpeg", "png"],
        key="agribot_img",
        label_visibility="collapsed",
    )

    # Track whether this specific image was already processed
    if "_last_img_id" not in st.session_state:
        st.session_state._last_img_id = None

    # Determine if this is a NEW image (not already processed)
    new_image = False
    if uploaded_img is not None:
        img_id = (uploaded_img.name, uploaded_img.size)
        if img_id != st.session_state._last_img_id:
            new_image = True

    # ── Chat input ──
    user_input = st.chat_input("Describe a crop issue, ask for advice, or upload an image...")

    if user_input or new_image:
        # Build the user message text
        display_text = user_input if user_input else "Please analyze this crop image."

        # Show user message
        with st.chat_message("user", avatar="👤"):
            if uploaded_img:
                st.image(uploaded_img, width=250)
            st.markdown(display_text)

        # Save user message
        user_msg = {"role": "user", "content": display_text}
        if uploaded_img:
            user_msg["image"] = uploaded_img.getvalue()
        st.session_state.agribot_messages.append(user_msg)

        # ── Run trained model on uploaded image for disease detection ──
        model_diagnosis_text = ""
        model_diagnosis_card = ""
        if uploaded_img:
            # First validate it's actually a leaf
            is_leaf = validate_is_leaf(uploaded_img)
            if not is_leaf:
                model_diagnosis_text = (
                    "[MODEL NOTE: The uploaded image does not appear to contain a plant leaf. "
                    "Inform the user that this doesn't look like a crop leaf and suggest uploading a clear leaf image.]"
                )
            else:
                try:
                    uploaded_img.seek(0)
                    model = load_model()
                    pred_index, confidence, _ = predict_image(model, uploaded_img)
                    result = format_result(pred_index, confidence)
                    conf_pct = result['confidence'] * 100

                    if confidence >= CONFIDENCE_THRESHOLD:
                        if result['is_healthy']:
                            model_diagnosis_text = (
                                f"[MODEL DIAGNOSIS: The uploaded image is identified as a **{result['plant']}** plant. "
                                f"Status: **HEALTHY** (Confidence: {conf_pct:.1f}%). "
                                f"Provide tips to keep this plant healthy and prevent common diseases.]"
                            )
                            model_diagnosis_card = (
                                f'<div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-left:5px solid #2e7d32;'
                                f'border-radius:12px;padding:20px 24px;margin:0 0 16px 0;">'
                                f'<h3 style="color:#2e7d32;margin:0 0 4px 0;">✅ {result["plant"]} — Healthy</h3>'
                                f'<p style="color:#555;margin:0;">Confidence: <b>{conf_pct:.1f}%</b></p>'
                                f'<p style="color:#2e7d32;font-size:0.95rem;margin:8px 0 0 0;">🎉 Your plant looks healthy!</p>'
                                f'</div>'
                            )
                        else:
                            treatment = DISEASE_TREATMENT_DB.get(result['raw_class'], None)
                            model_diagnosis_text = (
                                f"[MODEL DIAGNOSIS: The uploaded image is identified as a **{result['plant']}** plant "
                                f"with **{result['disease']}** disease (Confidence: {conf_pct:.1f}%). "
                            )
                            if treatment:
                                model_diagnosis_text += (
                                    f"Cause: {treatment['cause']} "
                                    f"Recommended treatment: {treatment['treatment']}. "
                                    f"Dosage: {treatment['dosage']}. "
                                    f"Application: {treatment['application']}. "
                                )
                            model_diagnosis_text += (
                                f"Based on this diagnosis provide detailed treatment advice, "
                                f"including organic and chemical options, and prevention tips.]"
                            )

                            # Build card HTML
                            card_rows = ""
                            if treatment:
                                card_rows = (
                                    f'<div style="margin-top:12px;">'
                                    f'<p style="margin:6px 0;"><b style="color:#4e342e;">🔍 Cause:</b> {treatment["cause"]}</p>'
                                    f'<p style="margin:6px 0;"><b style="color:#2e7d32;">💊 Treatment:</b> {treatment["treatment"]}</p>'
                                    f'<p style="margin:6px 0;"><b style="color:#1565c0;">📏 Dosage:</b> {treatment["dosage"]}</p>'
                                    f'<p style="margin:6px 0;"><b style="color:#6a1b9a;">🌿 Application:</b> {treatment["application"]}</p>'
                                    f'</div>'
                                )
                            model_diagnosis_card = (
                                f'<div style="background:linear-gradient(135deg,#fff8e1,#fff3e0);border-left:5px solid #e65100;'
                                f'border-radius:12px;padding:20px 24px;margin:0 0 16px 0;'
                                f'box-shadow:0 2px 12px rgba(0,0,0,0.07);">'
                                f'<h3 style="color:#bf360c;margin:0 0 4px 0;">🦠 {result["plant"]} — {result["disease"]}</h3>'
                                f'<p style="color:#6d4c41;margin:0;">Confidence: <b>{conf_pct:.1f}%</b></p>'
                                f'{card_rows}'
                                f'</div>'
                            )
                    else:
                        model_diagnosis_text = (
                            "[MODEL NOTE: The uploaded image could not be confidently identified as a known crop leaf. "
                            "Analyze the image visually and provide your best assessment.]"
                        )
                except Exception:
                    pass  # Fall back to Gemini-only analysis

        # Build Gemini API payload with model diagnosis as context
        query_with_diagnosis = display_text
        if model_diagnosis_text:
            query_with_diagnosis = f"{model_diagnosis_text}\n\nUser message: {display_text}"

        parts = [{"text": f"[System Persona/Instructions: {AGRIBOT_SYSTEM_PROMPT}]\n\n--- User Query ---\n{query_with_diagnosis}"}]

        if uploaded_img:
            uploaded_img.seek(0)
            img_bytes = uploaded_img.read()
            img_b64 = _b64.b64encode(img_bytes).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": uploaded_img.type,
                    "data": img_b64,
                }
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }

        # Call Gemini API
        with st.chat_message("assistant", avatar="🌿"):
            # Show the model diagnosis card first (if available)
            if model_diagnosis_card:
                st.markdown(model_diagnosis_card, unsafe_allow_html=True)
                st.markdown("---")

            with st.spinner("🌱 AgriBot is thinking..."):
                try:
                    resp = _req_lib.post(
                        _GEMINI_API_URL,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        bot_text = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "I'm sorry, I couldn't process that.")
                        )
                    else:
                        bot_text = f"⚠️ API Error ({resp.status_code}): {resp.text[:200]}"
                except Exception as e:
                    bot_text = f"⚠️ Connection error: {str(e)}"

            st.markdown(bot_text)

        # Store both diagnosis card and AI text for chat history
        full_response = ""
        if model_diagnosis_card:
            full_response = model_diagnosis_card + "\n\n---\n\n"
        full_response += bot_text
        st.session_state.agribot_messages.append({"role": "assistant", "content": full_response})

        # Mark this image as processed so it doesn't re-trigger on rerun
        if uploaded_img is not None:
            st.session_state._last_img_id = (uploaded_img.name, uploaded_img.size)

        st.rerun()


# ─────────── FERTILIZER RECOMMENDATION ───────────
elif app_mode == "FERTILIZER RECOMMENDATION":

    # ── Fertilizer Database ──
    FERTILIZER_DB = {
        'Rice': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'Medium'},
            'fertilizer': 'NPK 20-20-20 or Urea + DAP',
            'dosage': 'Urea: 80–100 kg/acre, DAP: 50 kg/acre, MOP: 30 kg/acre',
            'timing': 'Basal dose at transplanting; top dress N at tillering and panicle initiation stages.',
            'tips': 'Apply in split doses. Use zinc sulfate (10 kg/acre) in zinc-deficient soils.',
        },
        'Wheat': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'Low'},
            'fertilizer': 'NPK 12-32-16 + Urea top dress',
            'dosage': 'Urea: 100–120 kg/acre, DAP: 50 kg/acre',
            'timing': 'Half N + full P & K at sowing; remaining N at first irrigation (21 days).',
            'tips': 'Apply sulfur (20 kg/acre) for better grain quality in sulfur-deficient soils.',
        },
        'Corn / Maize': {
            'nutrients': {'Nitrogen (N)': 'Very High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'Medium'},
            'fertilizer': 'NPK 20-20-20 + Urea side dress',
            'dosage': 'Urea: 120–150 kg/acre, DAP: 50 kg/acre, MOP: 40 kg/acre',
            'timing': 'Basal dose at sowing; side dress N at knee-high and tasseling stages.',
            'tips': 'Zinc sulfate recommended in alkaline soils. Consider foliar micronutrient spray.',
        },
        'Tomato': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'High', 'Potassium (K)': 'Very High'},
            'fertilizer': 'NPK 19-19-19 + Calcium Nitrate',
            'dosage': 'NPK: 100 kg/acre, Calcium Nitrate: 25 kg/acre',
            'timing': 'Basal NPK at transplanting; calcium nitrate during fruiting every 15 days.',
            'tips': 'High K improves fruit quality and color. Add boron for better fruit set.',
        },
        'Potato': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'High', 'Potassium (K)': 'Very High'},
            'fertilizer': 'NPK 12-32-16 + MOP (Muriate of Potash)',
            'dosage': 'DAP: 75 kg/acre, MOP: 60 kg/acre, Urea: 80 kg/acre',
            'timing': 'Full P & K at planting; N in 2 splits — at planting and earthing up.',
            'tips': 'Avoid excess nitrogen — it promotes foliage over tubers. Use sulfate of potash for quality.',
        },
        'Apple': {
            'nutrients': {'Nitrogen (N)': 'Medium', 'Phosphorus (P)': 'Low', 'Potassium (K)': 'High'},
            'fertilizer': 'NPK 15-15-15 + Calcium Ammonium Nitrate (CAN)',
            'dosage': 'NPK: 500–800 g/tree, CAN: 250–500 g/tree (varies by age)',
            'timing': 'First dose at bud break (Feb-Mar); second dose after fruit set.',
            'tips': 'Apply borax (10 g/tree) for better pollination. Foliar spray of zinc sulfate in spring.',
        },
        'Grape': {
            'nutrients': {'Nitrogen (N)': 'Medium', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'Very High'},
            'fertilizer': 'NPK 13-0-45 (Potassium Nitrate) + DAP',
            'dosage': 'KNO3: 5 g/L foliar spray, DAP: 100 kg/acre soil application',
            'timing': 'N & P after pruning; K during berry development and veraison.',
            'tips': 'Potassium is critical for sugar content and color. Avoid excess N during fruiting.',
        },
        'Pepper / Chili': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'High'},
            'fertilizer': 'NPK 19-19-19 + Urea side dress',
            'dosage': 'NPK: 80 kg/acre, Urea: 50 kg/acre in 2 splits',
            'timing': 'Basal at transplanting; top dress at flowering and fruiting stages.',
            'tips': 'Calcium spray prevents blossom end rot. Magnesium sulfate improves pod quality.',
        },
        'Cotton': {
            'nutrients': {'Nitrogen (N)': 'Very High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'High'},
            'fertilizer': 'Urea + SSP + MOP',
            'dosage': 'Urea: 130 kg/acre, SSP: 150 kg/acre, MOP: 40 kg/acre',
            'timing': 'Basal P & K at sowing; N in 3 splits — sowing, squaring, and boll formation.',
            'tips': 'Foliar application of 2% DAP at flowering boosts boll retention.',
        },
        'Sugarcane': {
            'nutrients': {'Nitrogen (N)': 'Very High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'High'},
            'fertilizer': 'Urea + SSP + MOP',
            'dosage': 'Urea: 150 kg/acre, SSP: 150 kg/acre, MOP: 60 kg/acre',
            'timing': 'Full P & K at planting; N in 3 equal splits at 30, 60, and 90 days.',
            'tips': 'Trash mulching recycles potassium. Apply iron sulfate in calcareous soils.',
        },
        'Soybean': {
            'nutrients': {'Nitrogen (N)': 'Low (fixes own N)', 'Phosphorus (P)': 'High', 'Potassium (K)': 'Medium'},
            'fertilizer': 'DAP + MOP (no Urea needed)',
            'dosage': 'DAP: 50 kg/acre, MOP: 20 kg/acre',
            'timing': 'Full dose at sowing as basal application.',
            'tips': 'Use Rhizobium seed inoculant for biological nitrogen fixation. Sulfur (20 kg/acre) improves oil content.',
        },
        'Strawberry': {
            'nutrients': {'Nitrogen (N)': 'Medium', 'Phosphorus (P)': 'High', 'Potassium (K)': 'High'},
            'fertilizer': 'NPK 12-12-36 (water-soluble) + Calcium Nitrate',
            'dosage': '3–5 g/L through drip fertigation every week',
            'timing': 'Start 2 weeks after planting; increase K during fruiting.',
            'tips': 'Use drip fertigation for precise delivery. Avoid excess N — it delays fruiting.',
        },
        'Cherry': {
            'nutrients': {'Nitrogen (N)': 'Medium', 'Phosphorus (P)': 'Low', 'Potassium (K)': 'Medium'},
            'fertilizer': 'NPK 15-15-15 + Calcium Ammonium Nitrate',
            'dosage': 'NPK: 300–600 g/tree based on age',
            'timing': 'Apply in early spring before bud break.',
            'tips': 'Calcium sprays during fruitlet stage reduce cracking. Avoid late-season nitrogen.',
        },
        'Peach': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Low', 'Potassium (K)': 'Medium'},
            'fertilizer': 'Urea + NPK 15-15-15',
            'dosage': 'Urea: 200–400 g/tree, NPK: 300–500 g/tree (varies by age)',
            'timing': 'Apply 4–6 weeks before bloom and after harvest.',
            'tips': 'Excessive N causes soft fruit and poor storage quality. Zinc foliar spray recommended.',
        },
        'Orange / Citrus': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'High'},
            'fertilizer': 'NPK 15-15-15 + Urea + Micronutrient mix',
            'dosage': 'NPK: 500–1000 g/tree/year, split into 3 applications',
            'timing': 'Feb (pre-bloom), June (fruit set), Sept (fruit development).',
            'tips': 'Zinc, manganese, and iron foliar sprays are essential. Mulch to conserve moisture.',
        },
        'Banana': {
            'nutrients': {'Nitrogen (N)': 'Very High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'Very High'},
            'fertilizer': 'Urea + MOP + SSP',
            'dosage': 'Urea: 250 g/plant, SSP: 250 g/plant, MOP: 300 g/plant per crop cycle',
            'timing': 'N & K in 4–5 splits starting 1 month after planting up to bunch emergence.',
            'tips': 'Potassium is the most critical nutrient. Apply FYM (10 kg/plant) for soil health.',
        },
        'Onion': {
            'nutrients': {'Nitrogen (N)': 'Medium', 'Phosphorus (P)': 'High', 'Potassium (K)': 'Medium'},
            'fertilizer': 'NPK 19-19-19 + Sulphur',
            'dosage': 'NPK: 60 kg/acre, Sulphur: 20 kg/acre',
            'timing': 'Basal at transplanting; N top dress at 30 and 45 days.',
            'tips': 'Sulfur improves pungency and storage quality. Stop nitrogen 2 weeks before harvest.',
        },
        'Groundnut / Peanut': {
            'nutrients': {'Nitrogen (N)': 'Low (legume)', 'Phosphorus (P)': 'High', 'Potassium (K)': 'Low'},
            'fertilizer': 'SSP (Single Super Phosphate) + Gypsum',
            'dosage': 'SSP: 100 kg/acre, Gypsum: 200 kg/acre at flowering',
            'timing': 'SSP at sowing; Gypsum at 40–45 days (peg initiation).',
            'tips': 'Gypsum supplies calcium and sulfur for proper pod filling. Rhizobium inoculation recommended.',
        },
        'Tea': {
            'nutrients': {'Nitrogen (N)': 'Very High', 'Phosphorus (P)': 'Low', 'Potassium (K)': 'Medium'},
            'fertilizer': 'Ammonium Sulfate + MOP',
            'dosage': 'Ammonium Sulfate: 150–200 kg/acre/year, MOP: 40 kg/acre/year',
            'timing': 'N in 4 splits — March, May, July, September.',
            'tips': 'Tea prefers ammonium-N over nitrate-N. Maintain soil pH 4.5–5.5.',
        },
        'Coffee': {
            'nutrients': {'Nitrogen (N)': 'High', 'Phosphorus (P)': 'Medium', 'Potassium (K)': 'High'},
            'fertilizer': 'NPK 10-5-20 complex',
            'dosage': '300–500 g/plant/year depending on age and yield',
            'timing': 'Apply in 2 splits — pre-monsoon (April-May) and post-monsoon (Sept-Oct).',
            'tips': 'Shade-grown coffee needs less N. Apply lime if soil pH drops below 5.5.',
        },
    }

    # ── Custom CSS ──
    st.markdown("""
    <style>
        .fert-header {
            background: linear-gradient(135deg, #1a5632 0%, #2d8659 50%, #3a7d44 100%);
            padding: 40px 30px;
            border-radius: 14px;
            text-align: center;
            margin-bottom: 30px;
        }
        .fert-header h1 { color: #fff; margin: 0 0 6px 0; font-size: 2rem; }
        .fert-header p { color: rgba(255,255,255,0.85); margin: 0; font-size: 1.05rem; }
        .nutrient-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 4px;
        }
        .nutrient-high { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
        .nutrient-medium { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
        .nutrient-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
        .nutrient-vhigh { background: #fce4ec; color: #880e4f; border: 1px solid #f48fb1; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="fert-header">
        <h1>🧪 Fertilizer Recommendation</h1>
        <p>Select a crop to get tailored nutrient and fertilizer suggestions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Crop selector ──
    crop_list = sorted(FERTILIZER_DB.keys())
    selected_crop = st.selectbox("🌾 Select Your Crop", crop_list, index=None, placeholder="Choose a crop...")

    if selected_crop:
        info = FERTILIZER_DB[selected_crop]

        # ── Nutrient requirement badges ──
        def _badge_class(level):
            l = level.lower()
            if 'very high' in l: return 'nutrient-vhigh'
            if 'high' in l: return 'nutrient-high'
            if 'medium' in l: return 'nutrient-medium'
            return 'nutrient-low'

        st.markdown("### 📋 Recommendation")

        # ── Main card ──
        nutrient_badges = "".join(
            f'<span class="nutrient-badge {_badge_class(level)}">{name}: {level}</span>'
            for name, level in info['nutrients'].items()
        )

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa, #eef2f7); border-left: 5px solid #1a5632;
                    border-radius: 12px; padding: 28px 32px; margin: 10px 0;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
            <h2 style="color: #1a5632; margin: 0 0 8px 0;">🌾 {selected_crop}</h2>
            <hr style="border: none; border-top: 1px solid #c8e6c9; margin: 10px 0 18px 0;">

            <div style="margin-bottom: 18px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #4e342e;">🧬 Required Nutrients</span>
                <div style="margin-top: 8px;">{nutrient_badges}</div>
            </div>

            <div style="margin-bottom: 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #2e7d32;">💊 Suggested Fertilizer</span>
                <p style="color: #555; margin: 4px 0 0 0; font-size: 1.05rem;">• {info['fertilizer']}</p>
            </div>

            <div style="margin-bottom: 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #1565c0;">📏 Dosage</span>
                <p style="color: #555; margin: 4px 0 0 0;">• {info['dosage']}</p>
            </div>

            <div style="margin-bottom: 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #6a1b9a;">📅 Application Timing</span>
                <p style="color: #555; margin: 4px 0 0 0;">• {info['timing']}</p>
            </div>

            <div style="margin-bottom: 6px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #bf360c;">💡 Pro Tips</span>
                <p style="color: #555; margin: 4px 0 0 0;">• {info['tips']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── AI Deep-dive button ──
        st.markdown("---")
        if st.button("🤖 Get Detailed AI Advice", use_container_width=True):
            _fert_prompt = (
                f"As an expert agronomist, provide a detailed fertilizer and nutrient management plan for **{selected_crop}**.\n"
                f"Include:\n"
                f"1. **Soil Preparation** — How to prepare soil before applying fertilizer\n"
                f"2. **Stage-wise Fertilizer Schedule** — Exact timing and quantities at each growth stage\n"
                f"3. **Organic Alternatives** — Compost, vermicompost, bio-fertilizers\n"
                f"4. **Micronutrient Needs** — Zinc, Boron, Iron, Manganese recommendations\n"
                f"5. **Common Mistakes** — What farmers should avoid\n\n"
                f"Keep it practical, concise, and farmer-friendly with bullet points."
            )
            _fert_sys = "You are AgriBot, an expert AI Agronomist specializing in soil science and crop nutrition. Use markdown formatting."

            with st.spinner("🌱 AgriBot is preparing detailed advice..."):
                try:
                    _fert_payload = {
                        "contents": [{"role": "user", "parts": [
                            {"text": f"[System: {_fert_sys}]\n\n{_fert_prompt}"}
                        ]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 1500,
                        },
                    }
                    _fert_resp = _req_lib.post(
                        _GEMINI_API_URL,
                        json=_fert_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30,
                    )
                    if _fert_resp.status_code == 200:
                        _fert_data = _fert_resp.json()
                        _fert_advice = (
                            _fert_data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "Could not generate advice.")
                        )
                    else:
                        _fert_advice = f"⚠️ Could not fetch advice (API error {_fert_resp.status_code})."
                except Exception as _fe:
                    _fert_advice = f"⚠️ Connection error: {str(_fe)}"

            st.subheader("🌱 AgriBot Detailed Advice")
            st.markdown(_fert_advice)
    else:
        # Show placeholder when no crop selected
        st.info("👆 **Select a crop** from the dropdown above to see fertilizer recommendations.")
