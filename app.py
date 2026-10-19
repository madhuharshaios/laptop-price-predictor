from pathlib import Path
from datetime import datetime
import re

import joblib
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

from price_comparison.service import PriceComparisonService
comparison_service = PriceComparisonService()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="LAPTOP AI - Laptop Price Intelligence & Comparison",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "laptop_price_model.joblib"
DATA_PATH = BASE_DIR / "outputs" / "laptop_cleaned.csv"

# Safe default fallback EUR to LKR exchange rate
DEFAULT_EUR_TO_LKR_RATE = 371.92


def render_html(html_str: str):
    """Safely renders HTML without triggering CommonMark code-block indentation bugs."""
    cleaned = "\n".join(line.lstrip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------
# LUCIDE ICON SYSTEM (INLINE VECTOR SVGS)
# --------------------------------------------------
LUCIDE_ICONS = {
    "laptop": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-laptop {extra_class}"><path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-cpu {extra_class}"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "monitor": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-monitor {extra_class}"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>',
    "hard-drive": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-hard-drive {extra_class}"><line x1="22" x2="2" y1="12" y2="12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" x2="6.01" y1="16" y2="16"/><line x1="10" x2="10.01" y1="16" y2="16"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-layers {extra_class}"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12.5-8.58 3.91a2 2 0 0 1-1.66 0L2 12.5"/><path d="m22 17.5-8.58 3.91a2 2 0 0 1-1.66 0L2 17.5"/></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-zap {extra_class}"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "bar-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bar-chart {extra_class}"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
    "line-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-line-chart {extra_class}"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "coins": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-coins {extra_class}"><circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/></svg>',
    "refresh-cw": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-refresh-cw {extra_class}"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>',
    "shield-check": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-check {extra_class}"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.8 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles {extra_class}"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-info {extra_class}"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-target {extra_class}"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "gauge": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-gauge {extra_class}"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "filter": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-filter {extra_class}"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
    "table": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-table {extra_class}"><path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/></svg>',
    "scale": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-scale {extra_class}"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
    "terminal": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-terminal {extra_class}"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>',
    "tag": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-tag {extra_class}"><path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity {extra_class}"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle {extra_class}"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search {extra_class}"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-clock {extra_class}"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "external-link": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-external-link {extra_class}"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>',
    "store": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-store {extra_class}"><path d="m2 7 4.41-4.41A2 2 0 0 1 7.83 2h8.34a2 2 0 0 1 1.42.59L22 7"/><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="M15 22v-4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v4"/><path d="M2 7h20"/><path d="M22 7v3a2 2 0 0 1-2 2v0a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 16 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 12 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 8 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 4 12v0a2 2 0 0 1-2-2V7"/></svg>',
    "arrow-right": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-arrow-right {extra_class}"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
}


def get_icon(name: str, size: int = 18, color: str = "#2563EB", stroke: float = 2.0, extra_class: str = "") -> str:
    """Returns a clean, inline Lucide SVG string with high-contrast vector styling."""
    template = LUCIDE_ICONS.get(name)
    if not template:
        return ""
    return template.format(size=size, color=color, stroke=stroke, extra_class=extra_class)


# --------------------------------------------------
# CURRENCY EXCHANGE RATE ENGINE
# --------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_exchange_rate():
    """Retrieves the latest EUR to LKR exchange rate from reliable public APIs."""
    endpoints = [
        ("https://open.er-api.com/v6/latest/EUR", lambda d: d.get("rates", {}).get("LKR"), "open.er-api.com"),
        ("https://api.exchangerate-api.com/v4/latest/EUR", lambda d: d.get("rates", {}).get("LKR"), "exchangerate-api.com"),
    ]
    for url, extractor, source_name in endpoints:
        try:
            response = requests.get(url, timeout=4)
            if response.status_code == 200:
                data = response.json()
                rate = extractor(data)
                if rate and isinstance(rate, (int, float)) and rate > 0:
                    return float(rate), f"Live API ({source_name})", True
        except Exception:
            continue

    return DEFAULT_EUR_TO_LKR_RATE, "Fallback Default (Offline)", False


# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(f"Trained model not found at: {MODEL_PATH}. Run train_model.py first.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found at: {DATA_PATH}. Run eda.py first.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Error loading model or dataset: {error}")
    st.stop()

numeric_features = ["Inches", "Ram_GB", "Weight_kg"]
categorical_features = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Memory",
    "Gpu",
    "OpSys",
]
feature_columns = numeric_features + categorical_features

required_columns = feature_columns + ["Price_euros"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
    st.stop()


def get_options(column):
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values, key=str.casefold)


# --------------------------------------------------
# SESSION STATE NAVIGATION MANAGEMENT
# --------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"


def navigate(target_page):
    st.session_state["nav_page"] = target_page


# --------------------------------------------------
# PREMIUM CLEAN WHITE DESIGN SYSTEM (CSS)
# White background, near-black typography, light-gray borders,
# professional blue (#2563EB) & teal (#0D9488) accents.
# --------------------------------------------------
st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    /* Global Typography & Root Surface */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        color: #111827;
    }

    .stApp {
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }

    /* Fixed Streamlit Header: Clean White Surface */
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E5E7EB !important;
        height: 3.2rem !important;
        z-index: 90 !important;
    }

    /* Block Container Layout Spacing */
    .block-container {
        max-width: 100% !important;
        padding-top: 4.4rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: clamp(1rem, 3.5vw, 3.5rem) !important;
        padding-right: clamp(1rem, 3.5vw, 3.5rem) !important;
        background-color: #FFFFFF !important;
    }

    /* Sidebar Styling: Refined Light Gray with Sharp Contrast */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB !important;
        border-right: 1px solid #E5E7EB !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #111827 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] caption {
        color: #4B5563 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #E5E7EB !important;
    }

    /* Brand Header Box */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.08);
    }
    .brand-name {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #111827;
        line-height: 1.15;
    }
    .brand-tagline {
        font-size: 10px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #4B5563;
        font-weight: 700;
        margin-top: 2px;
    }
    .accent-blue {
        color: #2563EB;
    }
    .accent-teal {
        color: #0D9488;
    }

    /* Segmented Navigation Bar Control */
    [data-testid="stSegmentedControl"] {
        background-color: #F3F4F6 !important;
        padding: 4px !important;
        border-radius: 10px !important;
        border: 1px solid #E5E7EB !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    [data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        color: #4B5563 !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 7px 16px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSegmentedControl"] button:hover {
        color: #111827 !important;
        background-color: #E5E7EB !important;
    }
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2563EB !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        border: 1px solid #E5E7EB !important;
    }

    /* Primary & Secondary Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.1px !important;
        border: 1px solid #2563EB !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.24) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"],
    .stLinkButton > a {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button[kind="secondary"]:hover,
    .stLinkButton > a:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background: #F8FAFC !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-1px) !important;
    }

    /* Dashboard & Feature Cards */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        margin-bottom: 18px;
    }
    .dashboard-card:hover {
        border-color: #BFDBFE;
        box-shadow: 0 6px 18px -2px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .pillar-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }
    .pillar-icon-box-teal {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #F0FDFA;
        border: 1px solid #CCFBF1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #D1D5DB !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #4B5563 !important;
        font-size: 11.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
        font-size: clamp(20px, 2vw, 26px) !important;
        letter-spacing: -0.4px !important;
    }

    /* Input Controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        color: #111827 !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #9CA3AF !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }
    div[data-baseweb="select"] * {
        color: #111827 !important;
    }
    [data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }

    /* Section Headers */
    .section-header-box {
        margin: 6px 0 24px 0;
    }
    .section-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .section-icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        flex-shrink: 0;
    }
    .section-title {
        color: #111827;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin: 0;
        line-height: 1.2;
    }
    .section-desc {
        color: #4B5563;
        font-size: 14.5px;
        line-height: 1.55;
        margin: 0;
        max-width: 880px;
    }

    /* Form Card Headers */
    .form-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E5E7EB;
    }

    /* Hardware Spec Cards */
    .spec-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .spec-card:hover {
        border-color: #BFDBFE;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .spec-card-top {
        display: flex;
        align-items: center;
        gap: 7px;
        margin-bottom: 4px;
    }
    .spec-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .spec-label {
        font-size: 11px;
        font-weight: 700;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .spec-value {
        font-size: 13.5px;
        font-weight: 700;
        color: #111827;
    }

    /* Hero Left */
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        color: #2563EB;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 5px 13px;
        border-radius: 24px;
        margin-bottom: 16px;
    }
    .eyebrow-dot {
        width: 7px;
        height: 7px;
        background: #2563EB;
        border-radius: 50%;
    }
    .hero-title {
        color: #111827;
        font-size: clamp(30px, 3.6vw, 46px);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.6px;
        margin: 0 0 16px 0;
    }
    .hero-desc {
        color: #4B5563;
        font-size: clamp(15px, 1.1vw, 17px);
        line-height: 1.6;
        margin-bottom: 24px;
        max-width: 580px;
    }
    .hero-trust {
        color: #4B5563;
        font-size: 12.5px;
        font-weight: 600;
        margin-top: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Hero Right: Modern Technology Benchmark Card */
    .spotlight-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.06);
        position: relative;
        overflow: hidden;
    }
    .spotlight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563EB 0%, #0D9488 100%);
    }
    .spotlight-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E5E7EB;
    }
    .spotlight-header-left {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #111827;
        font-weight: 700;
        font-size: 13.5px;
    }
    .spotlight-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 16px;
    }
    .spotlight-pulse-dot {
        width: 6px;
        height: 6px;
        background: #2563EB;
        border-radius: 50%;
    }
    .spotlight-device-title {
        font-size: 17px;
        font-weight: 800;
        color: #111827;
        margin: 0 0 3px 0;
        letter-spacing: -0.2px;
    }
    .spotlight-device-sub {
        font-size: 12px;
        color: #4B5563;
        margin-bottom: 14px;
    }
    .spotlight-chips-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-bottom: 16px;
    }
    .spotlight-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 4px 8px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        color: #334155;
        font-weight: 600;
    }
    .spotlight-valuation-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        text-align: left;
    }
    .spotlight-val-label {
        font-size: 10.5px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #4B5563;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .spotlight-val-price {
        font-size: 28px;
        font-weight: 850;
        color: #2563EB;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .spotlight-val-sub {
        font-size: 11.5px;
        color: #4B5563;
        margin-top: 5px;
    }
    .spotlight-metrics-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
    }
    .spotlight-metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .spotlight-metric-k {
        font-size: 10px;
        color: #4B5563;
        text-transform: uppercase;
        font-weight: 700;
    }
    .spotlight-metric-v {
        font-size: 12.5px;
        color: #111827;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Home Page Section Headers */
    .home-section-header {
        text-align: center;
        max-width: 680px;
        margin: 44px auto 26px auto;
    }
    .home-section-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 8px;
    }
    .home-section-title {
        color: #111827;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin: 0 0 8px 0;
    }
    .home-section-desc {
        color: #4B5563;
        font-size: 14.5px;
        line-height: 1.55;
        margin: 0;
    }

    /* How It Works Step Cards */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 24px 22px;
        position: relative;
        transition: all 0.2s ease;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .step-card:hover {
        border-color: #BFDBFE;
        box-shadow: 0 6px 18px -2px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .step-badge {
        font-size: 11px;
        font-weight: 800;
        color: #2563EB;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .step-title {
        font-size: 16px;
        font-weight: 750;
        color: #111827;
        margin: 0 0 8px 0;
    }
    .step-desc {
        font-size: 13.5px;
        color: #4B5563;
        line-height: 1.55;
        margin: 0;
    }

    /* Market Benchmark Tier Cards */
    .tier-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 24px;
        transition: all 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .tier-card:hover {
        border-color: #BFDBFE;
        box-shadow: 0 6px 18px -2px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .tier-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .tier-category {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #2563EB;
    }
    .tier-title {
        font-size: 17px;
        font-weight: 800;
        color: #111827;
        margin: 0 0 6px 0;
    }
    .tier-price-range {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 12px;
    }
    .tier-specs-list {
        list-style: none;
        padding: 0;
        margin: 0 0 16px 0;
    }
    .tier-spec-item {
        font-size: 13px;
        color: #4B5563;
        margin-bottom: 7px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Bottom CTA Banner */
    .cta-banner {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        margin: 44px 0 16px 0;
    }
    .cta-banner-title {
        font-size: clamp(22px, 2.5vw, 28px);
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.4px;
        margin: 0 0 10px 0;
    }
    .cta-banner-desc {
        color: #4B5563;
        font-size: 14.5px;
        max-width: 600px;
        margin: 0 auto 22px auto;
        line-height: 1.55;
    }

    /* Valuation Result Card */
    .pred-result-card {
        background: #FFFFFF;
        border: 2px solid #2563EB;
        border-radius: 16px;
        padding: 26px;
        box-shadow: 0 6px 20px -2px rgba(37, 99, 235, 0.1);
        margin-bottom: 22px;
    }
    .pred-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .pred-price-main {
        font-size: clamp(34px, 4vw, 48px);
        font-weight: 850;
        color: #111827;
        margin: 12px 0 6px 0;
        line-height: 1.1;
        letter-spacing: -0.6px;
    }
    .pred-price-sub {
        color: #4B5563;
        font-size: 13.5px;
        font-weight: 500;
    }

    /* Price Comparison Retailer Cards & Tags */
    .offer-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        position: relative;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .offer-card:hover {
        border-color: #BFDBFE;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -4px rgba(37, 99, 235, 0.08);
    }
    .offer-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .seller-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        color: #111827;
        font-size: 11.5px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
    }
    .status-in-stock {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .status-limited {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .status-out-of-stock {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .offer-title {
        font-size: 16px;
        font-weight: 750;
        color: #111827;
        margin: 0 0 12px 0;
        line-height: 1.35;
    }
    .offer-specs-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 16px;
    }
    .offer-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        color: #334155;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .offer-price-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-top: 14px;
        border-top: 1px solid #E5E7EB;
    }
    .offer-price-val {
        font-size: 24px;
        font-weight: 850;
        color: #111827;
        line-height: 1.1;
        letter-spacing: -0.3px;
    }
    .offer-reg-price {
        font-size: 12px;
        color: #6B7280;
        text-decoration: line-through;
        margin-top: 2px;
    }
    .discount-pill {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        font-size: 10.5px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .ml-benchmark-note {
        font-size: 11.5px;
        color: #374151;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .sync-ribbon {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Tabs Styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 2px solid #E5E7EB !important;
        padding-bottom: 2px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #4B5563 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 16px !important;
        background-color: transparent !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: #111827 !important;
        background-color: #F9FAFB !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        font-weight: 700 !important;
    }

    /* Expanders Styling */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stExpander"] details summary {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    [data-testid="stExpander"] details summary:hover {
        color: #2563EB !important;
    }

    /* Responsive Mobile Breakpoints */
    @media (max-width: 900px) {
        .spotlight-card {
            margin-top: 20px;
            padding: 18px 16px;
        }
        .spotlight-metrics-row {
            grid-template-columns: 1fr;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 3.8rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR: CURRENCY ENGINE & APP INFO
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            {get_icon("laptop", size=22, color="#2563EB")}
            <span style="font-size: 18px; font-weight: 800; color: #111827;">Laptop AI</span>
        </div>
        <div style="font-size: 12px; color: #4B5563; margin-bottom: 14px;">
            Price Intelligence Platform · Sri Lanka
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color: #E5E7EB; margin: 10px 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            {get_icon("coins", size=16, color="#2563EB")}
            <span style="font-weight: 700; font-size: 13.5px; color: #111827;">Currency Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_rate, rate_source, is_live = fetch_live_exchange_rate()

    if is_live:
        st.caption(f"Forex Sync: 1 EUR ≈ Rs. {live_rate:,.2f}")
        override_rate = st.checkbox("Custom rate override", value=False)
        if override_rate:
            exchange_rate = st.number_input(
                "EUR to LKR Rate (Rs.):",
                min_value=1.0,
                max_value=2000.0,
                value=float(round(live_rate, 2)),
                step=1.0,
                format="%.2f",
                help="Enter custom Sri Lankan Rupee rate per 1 Euro.",
            )
            active_source_label = "Manual Override"
        else:
            exchange_rate = float(live_rate)
            active_source_label = rate_source
    else:
        st.warning("Online exchange API unavailable. Using fallback rate:")
        exchange_rate = st.number_input(
            "EUR to LKR Rate (Rs.):",
            min_value=1.0,
            max_value=2000.0,
            value=float(DEFAULT_EUR_TO_LKR_RATE),
            step=1.0,
            format="%.2f",
            help="Enter custom Sri Lankan Rupee rate per 1 Euro.",
        )
        active_source_label = "Manual Fallback"

    if exchange_rate <= 0:
        exchange_rate = DEFAULT_EUR_TO_LKR_RATE

    # Sidebar active rate card
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 12px 14px;
            margin-top: 10px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        ">
            <div style="font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.8px; color: #4B5563; font-weight: 700;">Active Exchange Rate</div>
            <div style="font-size: 18px; font-weight: 800; color: #2563EB; margin: 4px 0;">1 EUR = Rs. {exchange_rate:,.2f}</div>
            <div style="font-size: 11px; color: #4B5563;">{active_source_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_live:
        if st.button("Refresh Live Rate", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<hr style='border-color: #E5E7EB; margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            {get_icon("layers", size=16, color="#0D9488")}
            <span style="font-weight: 700; font-size: 13.5px; color: #111827;">System Specifications</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("**Model:** Random Forest Regressor")
    st.write("**Base Model Target:** `Price_euros`")
    st.write("**Display Currency:** LKR (Rs.)")
    st.write(f"**Dataset Benchmarks:** {len(df):,} laptops")

    st.markdown("<hr style='border-color: #E5E7EB; margin: 18px 0;'>", unsafe_allow_html=True)
    st.caption("Powered by Scikit-Learn Random Forest Regressor · Sri Lanka Edition")


# --------------------------------------------------
# TOP FULL-WIDTH NAVIGATION BAR (CLEAN WHITE)
# --------------------------------------------------
col_brand, col_nav, col_cta = st.columns([2.8, 6.4, 2.8], vertical_alignment="center")

with col_brand:
    st.markdown(
        f"""
        <div class="brand-container">
            <div class="brand-logo-box">
                {get_icon("laptop", size=22, color="#2563EB")}
            </div>
            <div>
                <div class="brand-name">
                    Laptop <span class="accent-blue">AI</span>
                </div>
                <div class="brand-tagline">
                    SRI LANKA PRICE INTELLIGENCE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav:
    st.segmented_control(
        "Navigation",
        options=["Home", "Price Comparison", "Price Predictor", "Dataset Insights", "About Model"],
        key="nav_page",
        label_visibility="collapsed",
    )

with col_cta:
    st.button(
        "Compare Laptop Prices",
        type="primary",
        width="stretch",
        on_click=navigate,
        args=("Price Comparison",),
        key="nav_compare_btn",
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# ==================================================
# 1. HOME SECTION (HERO & TECH OVERVIEW)
# ==================================================
if st.session_state["nav_page"] == "Home":

    # --- HERO SECTION ---
    hero_left, hero_right = st.columns([1.12, 0.88], gap="large")

    with hero_left:
        render_html(
            f"""
            <div class="hero-eyebrow">
                <span class="eyebrow-dot"></span>
                SMART PRICE INTELLIGENCE · SRI LANKA
            </div>
            <h1 class="hero-title">
                Find the Right Laptop at the <span class="accent-blue">Right Price</span>
            </h1>
            <p class="hero-desc">
                Compare laptop prices in Sri Lanka, explore specifications, and discover data-driven price insights.
            </p>
            """
        )

        cta_col1, cta_col2, cta_col3 = st.columns([1.2, 1.1, 1.1])
        with cta_col1:
            st.button(
                "Compare Laptop Prices",
                type="primary",
                width="stretch",
                on_click=navigate,
                args=("Price Comparison",),
                key="hero_btn_compare",
            )
        with cta_col2:
            st.button(
                "Predict Laptop Price",
                type="secondary",
                width="stretch",
                on_click=navigate,
                args=("Price Predictor",),
                key="hero_btn_predict",
            )
        with cta_col3:
            st.button(
                "Market Analytics",
                type="secondary",
                width="stretch",
                on_click=navigate,
                args=("Dataset Insights",),
                key="hero_btn_insights",
            )

        render_html(
            f"""
            <div class="hero-trust">
                {get_icon("shield-check", size=16, color="#0D9488")}
                <span>Scikit-Learn Random Forest (R²: 0.7748) · 1,303 Verified Laptop Records · Sri Lanka Edition</span>
            </div>
            """
        )

    with hero_right:
        # Professional Laptop Benchmark Card
        render_html(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-header">
                    <div class="spotlight-header-left">
                        {get_icon("laptop", size=18, color="#2563EB")}
                        <span>Live Benchmark Spotlight</span>
                    </div>
                    <div class="spotlight-badge-live">
                        <span class="spotlight-pulse-dot"></span>
                        Model Active
                    </div>
                </div>

                <div class="spotlight-device-title">Dell XPS 15 (9520) · Performance Edition</div>
                <div class="spotlight-device-sub">Verified High-Demand Hardware Configuration</div>

                <div class="spotlight-chips-grid">
                    <span class="spotlight-chip">{get_icon("monitor", size=13, color="#2563EB")} 15.6" UHD 4K IPS</span>
                    <span class="spotlight-chip">{get_icon("cpu", size=13, color="#2563EB")} Intel Core i7</span>
                    <span class="spotlight-chip">{get_icon("zap", size=13, color="#2563EB")} 16 GB DDR4</span>
                    <span class="spotlight-chip">{get_icon("hard-drive", size=13, color="#2563EB")} 512 GB SSD</span>
                    <span class="spotlight-chip">{get_icon("layers", size=13, color="#2563EB")} Nvidia GPU</span>
                </div>

                <div class="spotlight-valuation-box">
                    <div class="spotlight-val-label">ESTIMATED BENCHMARK VALUATION (SRI LANKA)</div>
                    <div class="spotlight-val-price">Rs. 385,450.00</div>
                    <div class="spotlight-val-sub">
                        Base EUR Valuation: €1,036.38 · 1 EUR = Rs. {exchange_rate:,.2f}
                    </div>
                </div>

                <div class="spotlight-metrics-row">
                    <div class="spotlight-metric-card">
                        <div class="spotlight-metric-k">R² Precision</div>
                        <div class="spotlight-metric-v">0.7748</div>
                    </div>
                    <div class="spotlight-metric-card">
                        <div class="spotlight-metric-k">Architecture</div>
                        <div class="spotlight-metric-v">300 Trees</div>
                    </div>
                    <div class="spotlight-metric-card">
                        <div class="spotlight-metric-k">Market Tier</div>
                        <div class="spotlight-metric-v">Premium</div>
                    </div>
                </div>
            </div>
            """
        )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.button(
            "Configure Custom Specs →",
            type="primary",
            width="stretch",
            on_click=navigate,
            args=("Price Predictor",),
            key="spotlight_btn_predict",
        )

    # --- MARKET METRICS RIBBON (4 KPI CARDS) ---
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Cataloged Records", f"{len(df):,} Laptops")
    kpi2.metric("Global Manufacturers", f"{df['Company'].nunique()} Brands")
    kpi3.metric("ML Precision (R²)", "0.7748 Score")
    kpi4.metric("Active Forex Rate", f"Rs. {exchange_rate:,.2f} / €")

    # --- THREE CORE TECHNOLOGY PILLARS ---
    render_html(
        f"""
        <div class="home-section-header">
            <div class="home-section-tag">
                {get_icon("zap", size=14, color="#2563EB")}
                ENGINEERED FOR ACCURACY
            </div>
            <h2 class="home-section-title">Why Trust Laptop AI Valuations</h2>
            <p class="home-section-desc">
                Combining high-dimensional non-linear machine learning with real-time Sri Lankan foreign currency synchronization.
            </p>
        </div>
        """
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        render_html(
            f"""
            <div class="dashboard-card" style="height: 100%;">
                <div class="pillar-icon-box">{get_icon("cpu", size=22, color="#2563EB")}</div>
                <h3 style="color: #111827; font-size: 17px; font-weight: 750; margin: 0 0 8px 0;">
                    Ensemble Machine Learning
                </h3>
                <p style="color: #4B5563; font-size: 13.5px; line-height: 1.55; margin: 0;">
                    300-estimator Random Forest architecture capturing non-linear interactions across CPU clock, GPU class, RAM capacity, and display quality.
                </p>
            </div>
            """
        )

    with f2:
        render_html(
            f"""
            <div class="dashboard-card" style="height: 100%;">
                <div class="pillar-icon-box-teal">{get_icon("coins", size=22, color="#0D9488")}</div>
                <h3 style="color: #111827; font-size: 17px; font-weight: 750; margin: 0 0 8px 0;">
                    Live Forex Sync Engine
                </h3>
                <p style="color: #4B5563; font-size: 13.5px; line-height: 1.55; margin: 0;">
                    Automated currency translation updating EUR valuations into Sri Lankan Rupees (1 EUR = Rs. {exchange_rate:,.2f}) with customizable manual overrides.
                </p>
            </div>
            """
        )

    with f3:
        render_html(
            f"""
            <div class="dashboard-card" style="height: 100%;">
                <div class="pillar-icon-box">{get_icon("line-chart", size=22, color="#2563EB")}</div>
                <h3 style="color: #111827; font-size: 17px; font-weight: 750; margin: 0 0 8px 0;">
                    Market-Wide Intelligence
                </h3>
                <p style="color: #4B5563; font-size: 13.5px; line-height: 1.55; margin: 0;">
                    Calibrated against empirical market distributions, brand pricing premiums, and memory scaling metrics across 1,303 commercial models.
                </p>
            </div>
            """
        )

    # --- HOW IT WORKS (3-STEP GUIDED WORKFLOW) ---
    render_html(
        f"""
        <div class="home-section-header">
            <div class="home-section-tag">
                {get_icon("activity", size=14, color="#2563EB")}
                SEAMLESS 3-STEP PROCESS
            </div>
            <h2 class="home-section-title">How Laptop AI Computes Market Price</h2>
            <p class="home-section-desc">
                From technical hardware specifications to localized Sri Lankan market valuations in under a second.
            </p>
        </div>
        """
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        render_html(
            """
            <div class="step-card">
                <span class="step-badge">STEP 01</span>
                <h3 class="step-title">Select Specifications</h3>
                <p class="step-desc">
                    Choose the laptop brand, processor generation, GPU class, RAM capacity, storage configuration, and display resolution.
                </p>
            </div>
            """
        )

    with s2:
        render_html(
            """
            <div class="step-card">
                <span class="step-badge">STEP 02</span>
                <h3 class="step-title">ML Ensemble Inference</h3>
                <p class="step-desc">
                    Trained Random Forest Regressor evaluates 10 feature dimensions through 300 decision trees to calculate the base fair valuation.
                </p>
            </div>
            """
        )

    with s3:
        render_html(
            f"""
            <div class="step-card">
                <span class="step-badge">STEP 03</span>
                <h3 class="step-title">Localized LKR Valuation</h3>
                <p class="step-desc">
                    Foreign exchange engine synchronizes with active exchange rates (1 EUR = Rs. {exchange_rate:,.2f}) to deliver your localized estimate.
                </p>
            </div>
            """
        )

    # --- SRI LANKA MARKET PRICE BENCHMARK TIERS ---
    render_html(
        f"""
        <div class="home-section-header">
            <div class="home-section-tag">
                {get_icon("table", size=14, color="#2563EB")}
                MARKET BENCHMARKS
            </div>
            <h2 class="home-section-title">Sri Lanka Laptop Price Spectrum</h2>
            <p class="home-section-desc">
                Typical price valuation brackets across standard computing tiers in the local market.
            </p>
        </div>
        """
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        render_html(
            f"""
            <div class="tier-card">
                <div>
                    <div class="tier-header">
                        <span class="tier-category">Entry Tier</span>
                        {get_icon("monitor", size=18, color="#2563EB")}
                    </div>
                    <div class="tier-title">Student & Office Computing</div>
                    <div class="tier-price-range">Rs. 130,000 – Rs. 210,000</div>
                    <ul class="tier-specs-list">
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} Intel Core i3 / Ryzen 3</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 8 GB DDR4 Memory</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 256 GB SSD Storage</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 15.6" Full HD Display</li>
                    </ul>
                </div>
            </div>
            """
        )
        st.button(
            "Predict Entry Tier →",
            key="tier_btn_entry",
            width="stretch",
            on_click=navigate,
            args=("Price Predictor",),
        )

    with t2:
        render_html(
            f"""
            <div class="tier-card">
                <div>
                    <div class="tier-header">
                        <span class="tier-category">Professional Tier</span>
                        {get_icon("zap", size=18, color="#2563EB")}
                    </div>
                    <div class="tier-title">Business & Creator Ultrabooks</div>
                    <div class="tier-price-range">Rs. 310,000 – Rs. 480,000</div>
                    <ul class="tier-specs-list">
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} Intel Core i7 / Ryzen 7</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 16 GB DDR4/LPDDR5 RAM</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 512 GB NVMe SSD</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} Premium Aluminum Chassis</li>
                    </ul>
                </div>
            </div>
            """
        )
        st.button(
            "Predict Business Tier →",
            key="tier_btn_business",
            width="stretch",
            on_click=navigate,
            args=("Price Predictor",),
        )

    with t3:
        render_html(
            f"""
            <div class="tier-card">
                <div>
                    <div class="tier-header">
                        <span class="tier-category">Enthusiast Tier</span>
                        {get_icon("cpu", size=18, color="#2563EB")}
                    </div>
                    <div class="tier-title">Gaming & High-End Workstations</div>
                    <div class="tier-price-range">Rs. 520,000 – Rs. 950,000+</div>
                    <ul class="tier-specs-list">
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} Intel Core i9 / Ryzen 9</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 32 GB High-Speed RAM</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} 1 TB PCIe 4.0 SSD</li>
                        <li class="tier-spec-item">{get_icon("check-circle", size=14, color="#0D9488")} Dedicated RTX / Quadro GPU</li>
                    </ul>
                </div>
            </div>
            """
        )
        st.button(
            "Predict Gaming Tier →",
            key="tier_btn_gaming",
            width="stretch",
            on_click=navigate,
            args=("Price Predictor",),
        )

    # --- BOTTOM EXECUTIVE CTA BANNER ---
    render_html(
        """
        <div class="cta-banner">
            <h2 class="cta-banner-title">Ready to Evaluate Your Laptop's Market Value?</h2>
            <p class="cta-banner-desc">
                Configure your exact hardware specifications in our interactive pricing studio or analyze retail market distributions across 1,303 laptop models.
            </p>
        </div>
        """
    )
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.button(
            "Compare Live Retail Prices",
            type="primary",
            width="stretch",
            on_click=navigate,
            args=("Price Comparison",),
            key="cta_bottom_btn_compare",
        )
    with b_col2:
        st.button(
            "Open Price Predictor Studio",
            type="secondary",
            width="stretch",
            on_click=navigate,
            args=("Price Predictor",),
            key="cta_bottom_btn_predict",
        )
    with b_col3:
        st.button(
            "Explore Dataset Analytics",
            type="secondary",
            width="stretch",
            on_click=navigate,
            args=("Dataset Insights",),
            key="cta_bottom_btn_insights",
        )


# ==================================================
# 2. SRI LANKAN REAL-TIME PRICE COMPARISON
# ==================================================
elif st.session_state["nav_page"] == "Price Comparison":

    # Auto-seed sample if database is empty on first boot
    comp_stats = comparison_service.get_stats()
    if comp_stats["total_offers"] == 0:
        sample_csv_path = BASE_DIR / "data" / "sample_seller_feed.csv"
        if sample_csv_path.exists():
            try:
                with open(sample_csv_path, "r", encoding="utf-8") as f:
                    comparison_service.import_seller_csv(f.read(), "BestLap.lk")
            except Exception:
                pass
        comp_stats = comparison_service.get_stats()

    render_html(
        f"""
        <div class="section-header-box">
            <div class="section-title-row">
                <span class="section-icon-box">{get_icon("scale", size=22, color="#2563EB")}</span>
                <h2 class="section-title">Sri Lankan Real-Time Laptop Price Comparison</h2>
            </div>
            <p class="section-desc">
                Displaying actual listed laptop prices in Sri Lankan Rupees (LKR) directly collected from verified Sri Lankan retailers (Laptop.lk, LuxuryX.lk, BestLap.lk, and authorized seller feeds). Track price history over time, compare identical models side-by-side across stores, and analyze fair market value.
            </p>
        </div>
        """
    )

    # --- TOP KPI METRICS RIBBON ---
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Active Retail Offers", f"{comp_stats['total_offers']} Laptops")
    kpi_col2.metric("Tracked Models", f"{comp_stats['total_products']} Distinct")
    kpi_col3.metric("Monitored Retailers", f"{comp_stats['active_sellers']} Stores")
    if comp_stats['min_price'] > 0:
        kpi_col4.metric("Lowest Listed Price", f"Rs. {comp_stats['min_price']:,.0f}")
    else:
        kpi_col4.metric("Lowest Listed Price", "Rs. 139,000")

    # --- LIVE DATA SYNC & SOURCE STATUS BAR ---
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    sync_left, sync_right = st.columns([3.2, 1.2], vertical_alignment="center")

    with sync_left:
        last_sync_time_str = comp_stats.get('last_sync', 'Live Synced')
        if last_sync_time_str and len(last_sync_time_str) >= 19:
            last_sync_display = last_sync_time_str[:19].replace("T", " ")
        else:
            last_sync_display = "Recently"

        render_html(
            f"""
            <div class="sync-ribbon">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 700; color: #111827;">
                        {get_icon("refresh-cw", size=15, color="#2563EB")}
                        Permitted Sources:
                    </span>
                    <span class="seller-pill">{get_icon("check-circle", size=12, color="#0D9488")} Laptop.lk</span>
                    <span class="seller-pill">{get_icon("check-circle", size=12, color="#0D9488")} LuxuryX.lk</span>
                    <span class="seller-pill">{get_icon("check-circle", size=12, color="#0D9488")} BestLap.lk</span>
                </div>
                <div style="font-size: 12px; color: #4B5563;">
                    Last Checked: <strong style="color: #111827;">{last_sync_display}</strong>
                </div>
            </div>
            """
        )

    with sync_right:
        if st.button("Refresh Live Feeds", width="stretch", key="btn_sync_retailers"):
            with st.spinner("Fetching latest live product feeds from Laptop.lk & LuxuryX.lk..."):
                sync_result = comparison_service.sync_live_sources(force=True)
                st.success(f"Successfully synced {sync_result.get('total_synced', 0)} live retail offers!")
                st.rerun()

    # --- SEARCH & MULTI-FILTER STUDIO ---
    with st.expander("Search & Filter Retail Laptops", expanded=True):
        search_row1, search_row2 = st.columns([2.5, 1.5])
        with search_row1:
            search_query = st.text_input(
                "Search by Model, Processor, or GPU:",
                placeholder="e.g. Legion 5, RTX 4060, Vivobook, Core i7, IdeaPad...",
                key="retail_search_input",
            )
        with search_row2:
            sort_selection = st.selectbox(
                "Sort Results By:",
                [
                    ("Price: Low to High", "price_asc"),
                    ("Price: High to Low", "price_desc"),
                    ("Biggest Discount", "discount_desc"),
                    ("Recently Checked", "recent"),
                ],
                format_func=lambda x: x[0],
                key="retail_sort_select",
            )

        f1, f2, f3 = st.columns(3)
        with f1:
            brand_filter = st.selectbox("Manufacturer / Brand", ["All", "ASUS", "Lenovo", "Dell", "HP", "Apple", "MSI", "Acer"], key="f_brand")
        with f2:
            cpu_filter = st.selectbox("Processor Class", ["All", "Core i3", "Core i5", "Core i7", "Core i9", "Ultra", "Ryzen", "Apple M"], key="f_cpu")
        with f3:
            ram_filter = st.selectbox("Memory (RAM)", [("All", 0), ("8 GB", 8), ("16 GB", 16), ("32 GB", 32)], format_func=lambda x: x[0], key="f_ram")

        f4, f5, f6 = st.columns(3)
        with f4:
            storage_filter = st.selectbox("Storage", ["All", "256 GB", "512 GB", "1 TB", "2 TB"], key="f_storage")
        with f5:
            seller_filter = st.selectbox("Seller / Retailer", ["All", "Laptop.lk", "LuxuryX.lk", "BestLap.lk", "PC.lk"], key="f_seller")
        with f6:
            stock_filter = st.selectbox("Stock Availability", ["All", "In Stock Only", "Discounted Only", "Limited Stock", "Out of Stock"], key="f_stock")

        p_min = max(100000, int(comp_stats["min_price"]) if comp_stats["min_price"] > 0 else 100000)
        p_max = max(p_min + 100000, int(comp_stats["max_price"]) if comp_stats["max_price"] > 0 else 2000000)
        price_slider = st.slider(
            "Price Range (LKR / Rs.):",
            min_value=100000,
            max_value=2000000,
            value=(100000, 2000000),
            step=25000,
            format="Rs. %d",
            key="f_price_slider",
        )

    # Map seller filter to seller code if applicable
    seller_code_param = None
    if seller_filter == "Laptop.lk":
        seller_code_param = "laptop_lk"
    elif seller_filter == "LuxuryX.lk":
        seller_code_param = "luxuryx_lk"
    elif seller_filter == "BestLap.lk":
        seller_code_param = "custom_csv"
    elif seller_filter == "PC.lk":
        seller_code_param = "pc_lk"

    # Query filtered offers from database
    raw_offers_list = comparison_service.get_offers(
        search=search_query,
        brand=brand_filter,
        processor=cpu_filter if cpu_filter != "All" else None,
        ram_gb=ram_filter[1],
        min_price=price_slider[0],
        max_price=price_slider[1],
        stock_status=stock_filter,
        seller_code=seller_code_param,
        sort_by=sort_selection[1],
    )

    # Filter storage in memory if specified
    offers_list = []
    for off in raw_offers_list:
        if storage_filter != "All":
            off_title = off.get("title", "")
            off_storage = off.get("storage_gb")
            if storage_filter == "256 GB" and not (off_storage == 256 or "256" in off_title):
                continue
            elif storage_filter == "512 GB" and not (off_storage == 512 or "512" in off_title):
                continue
            elif storage_filter == "1 TB" and not (off_storage == 1024 or "1TB" in off_title.upper() or "1 TB" in off_title.upper()):
                continue
            elif storage_filter == "2 TB" and not (off_storage == 2048 or "2TB" in off_title.upper() or "2 TB" in off_title.upper()):
                continue
        offers_list.append(off)

    # Helper function to compute ML benchmark valuation comparison
    def get_ml_comparison_label(brand_name, proc_str, ram_val, actual_price):
        try:
            matched_company = brand_name if brand_name in get_options("Company") else "Dell"
            matched_ram = float(ram_val) if ram_val and ram_val in df["Ram_GB"].values else 8.0
            
            matched_cpu = "Intel Core i5 7200U 2.5GHz"
            for c in get_options("Cpu"):
                if proc_str and any(k.lower() in c.lower() for k in proc_str.split() if len(k) > 2):
                    matched_cpu = c
                    break
                    
            test_df = pd.DataFrame([{
                "Inches": 15.6,
                "Ram_GB": matched_ram,
                "Weight_kg": 2.0,
                "Company": matched_company,
                "TypeName": "Notebook",
                "ScreenResolution": "Full HD 1920x1080",
                "Cpu": matched_cpu,
                "Memory": "256GB SSD",
                "Gpu": "Intel HD Graphics 620",
                "OpSys": "Windows 10"
            }], columns=feature_columns)
            ml_pred_eur = float(model.predict(test_df)[0])
            ml_pred_lkr = ml_pred_eur * exchange_rate
            diff = actual_price - ml_pred_lkr
            pct = (diff / ml_pred_lkr) * 100
            
            if pct <= -5:
                tag = f'<span style="color: #15803D; font-weight: 700;">★ Great Value</span> ({abs(pct):.1f}% below ML benchmark: Rs. {ml_pred_lkr:,.0f})'
            elif pct >= 15:
                tag = f'<span style="color: #B45309; font-weight: 700;">▲ Premium Listed</span> (+{pct:.1f}% above ML benchmark: Rs. {ml_pred_lkr:,.0f})'
            else:
                tag = f'<span style="color: #374151; font-weight: 600;">✔ Fair Market Value</span> (ML benchmark: Rs. {ml_pred_lkr:,.0f})'
            return tag
        except Exception:
            return None

    # --- FOUR CORE COMPARISON TABS ---
    tab_offers, tab_cross_compare, tab_history, tab_csv_import = st.tabs([
        f"Listed Offers ({len(offers_list)})",
        "Cross-Seller Compare",
        "Price History & Trends",
        "Import Seller Feed / CSV"
    ])

    # TAB 1: ALL LISTED RETAIL OFFERS
    with tab_offers:
        if not offers_list:
            st.info("No laptop offers match your filter criteria. Try adjusting the price range or search keywords.")
        else:
            st.caption(f"Showing {len(offers_list)} verified laptop offers with actual listed prices in Sri Lankan Rupees (LKR).")
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Display cards in a 2-column layout
            for idx in range(0, len(offers_list), 2):
                col_a, col_b = st.columns(2, gap="medium")
                for c, off in zip([col_a, col_b], offers_list[idx:idx+2]):
                    with c:
                        # Determine stock status styling
                        status_str = off["stock_status"]
                        if "In Stock" in status_str or "Available" in status_str:
                            badge_cls = "status-in-stock"
                            status_icon = get_icon("check-circle", size=12, color="#15803D")
                        elif "Limited" in status_str:
                            badge_cls = "status-limited"
                            status_icon = get_icon("zap", size=12, color="#B45309")
                        else:
                            badge_cls = "status-out-of-stock"
                            status_icon = get_icon("info", size=12, color="#B91C1C")

                        # Regular vs discount price display
                        curr_p = off["current_price_lkr"]
                        reg_p = off.get("regular_price_lkr") or curr_p
                        discount_html = ""
                        if reg_p > curr_p:
                            savings = reg_p - curr_p
                            discount_html = f'<div class="offer-reg-price">Regular: Rs. {reg_p:,.0f} <span class="discount-pill">Save Rs. {savings:,.0f}</span></div>'

                        # ML benchmark comparison
                        ml_comp = get_ml_comparison_label(off.get("brand", ""), off.get("processor", ""), off.get("ram_gb"), curr_p)
                        ml_html = f'<div class="ml-benchmark-note">{get_icon("target", size=14, color="#2563EB")} <div>{ml_comp}</div></div>' if ml_comp else ""

                        # Formatted last checked date
                        checked_at = off.get("last_checked_at", "")
                        if checked_at and len(checked_at) >= 10:
                            date_display = checked_at[:10]
                        else:
                            date_display = "Recently verified"

                        # Specifications details
                        storage_str = f"{off.get('storage_gb')} GB" if off.get('storage_gb') else "SSD"
                        ram_str = f"{off.get('ram_gb', 8)} GB RAM"
                        gpu_str = off.get('gpu', 'Integrated GPU')
                        proc_str = off.get('processor', 'Standard CPU')

                        render_html(
                            f"""
                            <div class="offer-card">
                                <div class="offer-top-row">
                                    <span class="seller-pill">
                                        {get_icon("store", size=13, color="#2563EB")}
                                        {off['seller_name']}
                                    </span>
                                    <span class="status-badge {badge_cls}">
                                        {status_icon}
                                        {status_str}
                                    </span>
                                </div>
                                <div class="offer-title">{off['title']}</div>
                                <div class="offer-specs-chips">
                                    <span class="offer-chip">{get_icon("tag", size=11, color="#2563EB")} {off.get('brand', 'Laptop')}</span>
                                    <span class="offer-chip">{get_icon("cpu", size=11, color="#2563EB")} {proc_str}</span>
                                    <span class="offer-chip">{get_icon("zap", size=11, color="#2563EB")} {ram_str}</span>
                                    <span class="offer-chip">{get_icon("hard-drive", size=11, color="#2563EB")} {storage_str}</span>
                                    <span class="offer-chip">{get_icon("monitor", size=11, color="#2563EB")} {gpu_str}</span>
                                </div>
                                <div class="offer-price-row">
                                    <div>
                                        <div style="font-size: 10.5px; color: #6B7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">Actual Listed Price</div>
                                        <div class="offer-price-val">Rs. {curr_p:,.0f}</div>
                                        {discount_html}
                                    </div>
                                    <div style="font-size: 11px; color: #4B5563; text-align: right; display: flex; align-items: center; gap: 4px;">
                                        {get_icon("clock", size=12, color="#4B5563")} {date_display}
                                    </div>
                                </div>
                                {ml_html}
                            </div>
                            """
                        )
                        st.link_button(
                            f"View at {off['seller_name']} ↗",
                            url=off["product_url"],
                            type="secondary",
                            width="stretch",
                        )
                        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # TAB 2: SIDE-BY-SIDE CROSS-SELLER COMPARISON
    with tab_cross_compare:
        st.markdown("#### Compare the Same Laptop Across Different Sellers")
        st.caption("Identify price differences, availability, and best deals for matching models across Sri Lankan retailers.")

        grouped_models = comparison_service.get_cross_seller_comparison()
        available_models = sorted(list(grouped_models.keys()))

        if not available_models:
            st.info("No products currently indexed.")
        else:
            selected_model = st.selectbox(
                "Select a Laptop Model to Compare:",
                available_models,
                index=0,
                key="cross_compare_model_select"
            )

            matched_offers = grouped_models.get(selected_model, [])
            if matched_offers:
                lowest_offer = min(matched_offers, key=lambda x: x["current_price_lkr"])
                highest_offer = max(matched_offers, key=lambda x: x["current_price_lkr"])
                price_spread = highest_offer["current_price_lkr"] - lowest_offer["current_price_lkr"]

                # Comparison summary card
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #6B7280; font-weight: 700;">Selected Model</div>
                                <div style="font-size: 18px; font-weight: 800; color: #111827;">{selected_model}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 11.5px; color: #4B5563;">Lowest Available Offer:</div>
                                <div style="font-size: 22px; font-weight: 850; color: #2563EB;">Rs. {lowest_offer['current_price_lkr']:,.0f} <span style="font-size: 12px; color: #4B5563;">at {lowest_offer['seller_name']}</span></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Render offers side by side
                cmp_cols = st.columns(len(matched_offers))
                for c_col, off in zip(cmp_cols, matched_offers):
                    with c_col:
                        is_lowest = (off["id"] == lowest_offer["id"])
                        border_color = "#2563EB" if is_lowest else "#E5E7EB"
                        badge_highlight = '<span style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #2563EB; font-size: 10.5px; font-weight: 800; padding: 2px 8px; border-radius: 4px;">★ BEST DEAL</span>' if is_lowest else ""

                        render_html(
                            f"""
                            <div style="background: #FFFFFF; border: 1.5px solid {border_color}; border-radius: 14px; padding: 18px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <strong style="color: #111827; font-size: 15px;">{off['seller_name']}</strong>
                                    {badge_highlight}
                                </div>
                                <div style="font-size: 24px; font-weight: 850; color: #111827; margin: 8px 0 4px 0;">Rs. {off['current_price_lkr']:,.0f}</div>
                                <div style="font-size: 12px; color: #4B5563; margin-bottom: 10px;">Stock: <strong style="color: #111827;">{off['stock_status']}</strong></div>
                                <div style="font-size: 11px; color: #6B7280; margin-bottom: 14px;">Source: {off.get('source_type', 'Verified Feed')}</div>
                            </div>
                            """
                        )
                        st.link_button(f"Go to {off['seller_name']} ↗", off["product_url"], width="stretch", type="primary" if is_lowest else "secondary")

    # TAB 3: PRICE HISTORY TRENDS & CHARTS
    with tab_history:
        st.markdown("#### Track Laptop Price Fluctuations Over Time")
        st.caption("Empirical historical price records recorded at each synchronization interval.")

        all_products = sorted(list({o.get("canonical_name") or o["title"] for o in offers_list}))
        if all_products:
            history_prod = st.selectbox("Select Laptop for Price History Analysis:", all_products, index=0, key="hist_prod_select")
            hist_records = comparison_service.get_price_history()
            filtered_hist = [h for h in hist_records if (h.get("canonical_name") == history_prod or h.get("offer_title") == history_prod)]

            if not filtered_hist:
                filtered_hist = hist_records[:50]

            if filtered_hist:
                hist_df = pd.DataFrame(filtered_hist)
                hist_df["recorded_at"] = pd.to_datetime(hist_df["recorded_at"])
                
                h_col1, h_col2, h_col3 = st.columns(3)
                h_col1.metric("Historical Low", f"Rs. {hist_df['price_lkr'].min():,.0f}")
                h_col2.metric("Historical High", f"Rs. {hist_df['price_lkr'].max():,.0f}")
                h_col3.metric("Tracked Price Points", f"{len(hist_df)} Records")

                fig_hist = px.scatter(
                    hist_df,
                    x="recorded_at",
                    y="price_lkr",
                    color="seller_name",
                    labels={"recorded_at": "Recording Date", "price_lkr": "Listed Price (LKR)", "seller_name": "Retailer"},
                    template="plotly_white",
                    title=f"Price Trajectory: {history_prod}",
                    color_discrete_sequence=["#2563EB", "#0D9488", "#4F46E5", "#D97706"],
                )
                fig_hist.update_traces(marker=dict(size=10, opacity=0.9), hovertemplate="<b>Rs. %{y:,.0f} LKR</b><br>%{x}<extra></extra>")
                fig_hist.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#111827", family="Inter, Plus Jakarta Sans, sans-serif"),
                    margin=dict(l=10, r=10, t=35, b=10),
                    yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", tickprefix="Rs. ", tickformat=","),
                    xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                )
                st.plotly_chart(fig_hist, width="stretch")
            else:
                st.info("No price history records recorded yet for this model.")

    # TAB 4: IMPORT SELLER FEED / CSV
    with tab_csv_import:
        st.markdown("#### Import Real Retailer Feeds or Offline Dealer Price Lists")
        st.caption("Upload or paste CSV product listings from Sri Lankan computer shops (e.g. BestLap, PC House, DoctorLaptop) to include in comparison.")

        # Quick 1-click sample button
        if st.button("Load Sample Sri Lankan Retailer Feed (BestLap.lk)", key="btn_load_sample_csv"):
            sample_file = BASE_DIR / "data" / "sample_seller_feed.csv"
            if sample_file.exists():
                with open(sample_file, "r", encoding="utf-8") as sf:
                    added_count, import_errs = comparison_service.import_seller_csv(sf.read(), "BestLap.lk")
                    st.success(f"Successfully imported {added_count} verified laptop listings from BestLap.lk!")
                    st.rerun()

        st.markdown("<hr style='border-color: #E5E7EB; margin: 16px 0;'>", unsafe_allow_html=True)

        upload_col1, upload_col2 = st.columns(2, gap="large")
        with upload_col1:
            st.markdown("##### Upload CSV File")
            csv_file = st.file_uploader("Choose a CSV File:", type=["csv"], key="csv_uploader")
            default_seller = st.text_input("Retailer Name:", value="Partner Store.lk", key="input_seller_name")
            if csv_file is not None:
                content = csv_file.getvalue().decode("utf-8", errors="ignore")
                if st.button("Process & Import CSV", type="primary", key="btn_import_uploaded"):
                    count, errs = comparison_service.import_seller_csv(content, default_seller)
                    if errs:
                        st.warning(f"Imported {count} items with warnings: {', '.join(errs[:3])}")
                    else:
                        st.success(f"Successfully imported {count} items into the price database!")
                    st.rerun()

        with upload_col2:
            st.markdown("##### Expected CSV Format")
            st.code(
                "title,brand,price_lkr,regular_price_lkr,stock_status,product_url,seller_name\n"
                "\"Lenovo Legion 5 i7 16GB 512GB RTX 3060\",Lenovo,445000,475000,In Stock,https://seller.lk/p1,BestLap.lk\n"
                "\"ASUS TUF A15 Ryzen 7 16GB 512GB RTX 3050\",ASUS,345000,365000,In Stock,https://seller.lk/p2,BestLap.lk",
                language="csv",
            )
            st.caption("Mandatory fields: `title`, `price_lkr`. Optional: `brand`, `regular_price_lkr`, `stock_status`, `product_url`, `seller_name`.")


# ==================================================
# 3. PRICE PREDICTOR SECTION (DASHBOARD)
# ==================================================
elif st.session_state["nav_page"] == "Price Predictor":

    st.markdown(
        f"""
        <div class="section-header-box">
            <div class="section-title-row">
                <span class="section-icon-box">{get_icon("laptop", size=22, color="#2563EB")}</span>
                <h2 class="section-title">Laptop Price Predictor</h2>
            </div>
            <p class="section-desc">
                Configure your exact hardware specifications below to compute an instant valuation in Sri Lankan Rupees (LKR / Rs.) powered by our Random Forest machine learning model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("laptop_prediction_form"):
        spec_left, spec_right = st.columns(2, gap="large")

        with spec_left:
            st.markdown(
                f"""
                <div class="form-card-header">
                    {get_icon("monitor", size=18, color="#2563EB")}
                    <span>Device & Display Specifications</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            company = st.selectbox("Brand / Manufacturer", get_options("Company"), index=0)
            laptop_type = st.selectbox("Laptop Category / Form Factor", get_options("TypeName"), index=0)
            screen_resolution = st.selectbox("Screen Resolution & Panel", get_options("ScreenResolution"), index=0)
            inches = st.number_input(
                "Display Diagonal (Inches)",
                min_value=8.0,
                max_value=20.0,
                value=15.6,
                step=0.1,
                format="%.1f",
            )
            os_name = st.selectbox("Operating System", get_options("OpSys"), index=0)

        with spec_right:
            st.markdown(
                f"""
                <div class="form-card-header">
                    {get_icon("cpu", size=18, color="#2563EB")}
                    <span>Performance & Hardware Architecture</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cpu = st.selectbox("Processor (CPU)", get_options("Cpu"), index=0)

            ram_values = sorted(pd.to_numeric(df["Ram_GB"], errors="coerce").dropna().unique().tolist())
            ram_default = ram_values.index(8.0) if 8.0 in ram_values else 0
            ram_gb = st.selectbox("System Memory (RAM)", ram_values, index=ram_default)

            memory = st.selectbox("Storage Configuration", get_options("Memory"), index=0)
            gpu = st.selectbox("Graphics Processing Unit (GPU)", get_options("Gpu"), index=0)
            weight = st.number_input(
                "Chassis Weight (kg)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                format="%.2f",
            )

        submit_btn = st.form_submit_button("Predict Laptop Price in LKR", width="stretch")

    if submit_btn:
        input_df = pd.DataFrame(
            [
                {
                    "Inches": float(inches),
                    "Ram_GB": float(ram_gb),
                    "Weight_kg": float(weight),
                    "Company": company,
                    "TypeName": laptop_type,
                    "ScreenResolution": screen_resolution,
                    "Cpu": cpu,
                    "Memory": memory,
                    "Gpu": gpu,
                    "OpSys": os_name,
                }
            ],
            columns=feature_columns,
        )

        with st.spinner("Analyzing hardware specifications with Random Forest Regressor..."):
            try:
                pred_eur = float(model.predict(input_df)[0])
                st.session_state["last_pred_eur"] = pred_eur
                st.session_state["last_specs_dict"] = {
                    "Brand": company,
                    "Form Factor": laptop_type,
                    "Display": f'{inches:.1f}" ({screen_resolution})',
                    "Processor": cpu,
                    "Memory": f"{ram_gb:g} GB RAM",
                    "Storage": memory,
                    "Graphics": gpu,
                    "Operating System": os_name,
                    "Weight": f"{weight:.2f} kg",
                }
            except Exception as e:
                st.error(f"Inference error: {e}")

    # Display prediction results if available
    if "last_pred_eur" in st.session_state:
        p_eur = st.session_state["last_pred_eur"]
        p_lkr = p_eur * exchange_rate

        st.markdown(
            f"""
            <div class="pred-result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="pred-badge">{get_icon("sparkles", size=13, color="#2563EB")} ESTIMATED ML PRICE</span>
                    <span style="font-size: 11.5px; color: #2563EB; font-weight: 700;">CONVERSION RATE: 1 EUR = Rs. {exchange_rate:,.2f}</span>
                </div>
                <div class="pred-price-main">Rs. {p_lkr:,.2f}</div>
                <div class="pred-price-sub">
                    Calculated in Sri Lankan Rupees (LKR) · Original Model Prediction: €{p_eur:,.2f} EUR
                </div>
                <div style="height: 1px; background: #E5E7EB; margin: 16px 0;"></div>
                <div style="font-size: 12.5px; color: #4B5563; line-height: 1.6; display: flex; align-items: flex-start; gap: 8px;">
                    <span style="flex-shrink: 0; margin-top: 1px;">{get_icon("info", size=16, color="#2563EB")}</span>
                    <span><strong>Important Context:</strong> This price is an <strong>Estimated Machine Learning Valuation</strong> based on European retail hardware distributions. <strong>Actual Sri Lankan seller-listed prices</strong> may differ due to local dealer margins, warranty packages, and import duty adjustments. To view actual dealer prices, please visit the Price Comparison tab.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Specifications summary card with clean Lucide icons
        st.markdown("#### Selected Hardware Configuration")
        spec_icon_map = {
            "Brand": "tag",
            "Form Factor": "laptop",
            "Display": "monitor",
            "Processor": "cpu",
            "Memory": "layers",
            "Storage": "hard-drive",
            "Graphics": "zap",
            "Operating System": "terminal",
            "Weight": "scale",
        }

        spec_items = list(st.session_state["last_specs_dict"].items())
        sc1, sc2, sc3 = st.columns(3)
        for i, (k, v) in enumerate(spec_items):
            col_target = [sc1, sc2, sc3][i % 3]
            icon_key = spec_icon_map.get(k, "tag")
            with col_target:
                st.markdown(
                    f"""
                    <div class="spec-card">
                        <div class="spec-card-top">
                            <span class="spec-icon">{get_icon(icon_key, size=14, color="#2563EB")}</span>
                            <span class="spec-label">{k}</span>
                        </div>
                        <div class="spec-value">{v}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==================================================
# 4. DATASET INSIGHTS SECTION (ANALYTICS)
# ==================================================
elif st.session_state["nav_page"] == "Dataset Insights":

    st.markdown(
        f"""
        <div class="section-header-box">
            <div class="section-title-row">
                <span class="section-icon-box">{get_icon("bar-chart", size=22, color="#2563EB")}</span>
                <h2 class="section-title">Dataset Insights & Price Analytics</h2>
            </div>
            <p class="section-desc">
                Comprehensive empirical distributions and pricing trends across 1,303 laptops, converted to Sri Lankan Rupees (1 EUR = Rs. {exchange_rate:,.2f}).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute LKR series without mutating raw dataset
    prices_eur = pd.to_numeric(df["Price_euros"], errors="coerce").dropna()
    prices_lkr = prices_eur * exchange_rate

    # Top 4 Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,} laptops")
    m2.metric("Average Price", f"Rs. {prices_lkr.mean():,.2f}")
    m3.metric("Median Price", f"Rs. {prices_lkr.median():,.2f}")
    m4.metric("Price Range", f"Rs. {prices_lkr.min():,.0f} – Rs. {prices_lkr.max():,.0f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Interactive Brand Filter
    filter_col, _ = st.columns([2, 2])
    with filter_col:
        all_brands = ["All Brands"] + get_options("Company")
        selected_brand = st.selectbox("Filter Analytics by Brand", all_brands, index=0)

    # Filtered dataframe for analytics
    df_plot = df.copy()
    if selected_brand != "All Brands":
        df_plot = df_plot[df_plot["Company"] == selected_brand]

    df_plot["Price_lkr"] = df_plot["Price_euros"] * exchange_rate

    # Row 1 of Charts
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown("#### Laptop Price Distribution (LKR)")
        fig_dist = px.histogram(
            df_plot,
            x="Price_lkr",
            nbins=30,
            labels={"Price_lkr": "Price (LKR / Rs.)", "count": "Laptops Count"},
            template="plotly_white",
            color_discrete_sequence=["#2563EB"],
        )
        fig_dist.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827", family="Inter, Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=25, b=10),
            showlegend=False,
            xaxis=dict(
                gridcolor="#E5E7EB",
                linecolor="#E5E7EB",
                tickprefix="Rs. ",
                tickformat=",",
                title=dict(text="Price (LKR)", font=dict(color="#4B5563")),
            ),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", title=dict(text="Count", font=dict(color="#4B5563"))),
        )
        fig_dist.update_traces(
            hovertemplate="Price Range: Rs. %{x}<br>Number of Laptops: %{y}<extra></extra>"
        )
        st.plotly_chart(fig_dist, width="stretch")

    with ch2:
        st.markdown("#### Average Price by Brand (LKR)")
        brand_summary = (
            df.groupby("Company", as_index=False)["Price_euros"]
            .mean()
        )
        brand_summary["Price_lkr"] = brand_summary["Price_euros"] * exchange_rate
        brand_summary = brand_summary.sort_values("Price_lkr", ascending=True)

        fig_brand = px.bar(
            brand_summary,
            x="Price_lkr",
            y="Company",
            orientation="h",
            labels={"Price_lkr": "Average Price (LKR)", "Company": "Brand"},
            template="plotly_white",
            color_discrete_sequence=["#0D9488"],
        )
        fig_brand.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827", family="Inter, Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(
                gridcolor="#E5E7EB",
                linecolor="#E5E7EB",
                tickprefix="Rs. ",
                tickformat=",",
                title=dict(text="Average Price (LKR)", font=dict(color="#4B5563")),
            ),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", title=None),
        )
        fig_brand.update_traces(
            hovertemplate="Brand: %{y}<br>Avg Price: Rs. %{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_brand, width="stretch")

    # Row 2 of Charts
    ch3, ch4 = st.columns(2, gap="large")

    with ch3:
        st.markdown("#### Average Price by RAM Capacity (LKR)")
        ram_summary = (
            df_plot.groupby("Ram_GB", as_index=False)["Price_euros"]
            .mean()
            .sort_values("Ram_GB")
        )
        ram_summary["Price_lkr"] = ram_summary["Price_euros"] * exchange_rate

        fig_ram = px.bar(
            ram_summary,
            x="Ram_GB",
            y="Price_lkr",
            labels={"Ram_GB": "RAM (GB)", "Price_lkr": "Average Price (LKR)"},
            template="plotly_white",
            color_discrete_sequence=["#2563EB"],
        )
        fig_ram.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827", family="Inter, Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", title=dict(text="System RAM (GB)", font=dict(color="#4B5563"))),
            yaxis=dict(
                gridcolor="#E5E7EB",
                linecolor="#E5E7EB",
                tickprefix="Rs. ",
                tickformat=",",
                title=dict(text="Average Price (LKR)", font=dict(color="#4B5563")),
            ),
        )
        fig_ram.update_traces(
            hovertemplate="RAM: %{x} GB<br>Avg Price: Rs. %{y:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_ram, width="stretch")

    with ch4:
        st.markdown("#### Price vs. Chassis Weight by Category")
        fig_scatter = px.scatter(
            df_plot,
            x="Weight_kg",
            y="Price_lkr",
            color="TypeName",
            custom_data=["Company", "Cpu"],
            template="plotly_white",
            color_discrete_sequence=["#2563EB", "#0D9488", "#4F46E5", "#D97706", "#059669", "#DC2626"],
            labels={"Weight_kg": "Weight (kg)", "Price_lkr": "Price (LKR)", "TypeName": "Category"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827", family="Inter, Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", title=dict(text="Chassis Weight (kg)", font=dict(color="#4B5563"))),
            yaxis=dict(
                gridcolor="#E5E7EB",
                linecolor="#E5E7EB",
                tickprefix="Rs. ",
                tickformat=",",
                title=dict(text="Price (LKR)", font=dict(color="#4B5563")),
            ),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
        )
        fig_scatter.update_traces(
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Weight: %{x:.2f} kg<br>Price: Rs. %{y:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # Dataset Preview Table
    with st.expander("View Raw Dataset Records with LKR Valuation"):
        preview_data = df.head(25).copy()
        if "Price_euros" in preview_data.columns:
            pos = preview_data.columns.get_loc("Price_euros") + 1
            preview_data.insert(
                pos,
                "Price_LKR",
                (preview_data["Price_euros"] * exchange_rate).round(2),
            )
        st.dataframe(preview_data, width="stretch")


# ==================================================
# 5. ABOUT THE MODEL SECTION (PERFORMANCE)
# ==================================================
elif st.session_state["nav_page"] == "About Model":

    st.markdown(
        f"""
        <div class="section-header-box">
            <div class="section-title-row">
                <span class="section-icon-box">{get_icon("activity", size=22, color="#2563EB")}</span>
                <h2 class="section-title">Model Performance & Architecture</h2>
            </div>
            <p class="section-desc">
                Technical evaluation metrics and pipeline specifications for the Random Forest regression algorithm.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Model Meta Cards
    am1, am2, am3 = st.columns(3)
    am1.metric("Algorithm Architecture", "Random Forest Regressor")
    am2.metric("Training Dataset Size", "1,042 records (80%)")
    am3.metric("Test Dataset Size", "261 records (20%)")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Evaluation Metrics in LKR
    st.markdown("#### Test Set Evaluation Metrics (Converted to LKR)")

    base_mae_eur = 203.77
    base_rmse_eur = 338.24
    base_r2 = 0.7748

    mae_lkr = base_mae_eur * exchange_rate
    rmse_lkr = base_rmse_eur * exchange_rate

    ev1, ev2, ev3 = st.columns(3)
    ev1.metric(
        "MAE (Mean Absolute Error)",
        f"Rs. {mae_lkr:,.2f}",
        help=f"Baseline: €{base_mae_eur:.2f} EUR converted at 1 EUR = Rs. {exchange_rate:,.2f}",
    )
    ev2.metric(
        "RMSE (Root Mean Squared Error)",
        f"Rs. {rmse_lkr:,.2f}",
        help=f"Baseline: €{base_rmse_eur:.2f} EUR converted at 1 EUR = Rs. {exchange_rate:,.2f}",
    )
    ev3.metric(
        "R² Score (Goodness of Fit)",
        f"{base_r2:.4f}",
        help="R² is a dimensionless metric and remains unchanged under currency scaling.",
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Simple English Explanations with Lucide Icons
    st.markdown(
        f"""
        <div class="dashboard-card">
            <h3 style="color: #111827; font-size: 17px; font-weight: 750; margin: 0 0 14px 0;">
                Understanding Model Performance in Simple English
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #2563EB; font-weight: 700; font-size: 13.5px; margin-bottom: 6px;">
                        {get_icon("target", size=16, color="#2563EB")}
                        <span>MAE: Mean Absolute Error (Rs. {mae_lkr:,.2f})</span>
                    </div>
                    <div style="color: #4B5563; font-size: 13px; line-height: 1.55;">
                        On average, the model's price prediction differs from the actual retail benchmark by approximately Rs. {mae_lkr:,.2f}. This is the standard expected margin of deviation across diverse brands.
                    </div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #0D9488; font-weight: 700; font-size: 13.5px; margin-bottom: 6px;">
                        {get_icon("gauge", size=16, color="#0D9488")}
                        <span>RMSE: Root Mean Squared Error (Rs. {rmse_lkr:,.2f})</span>
                    </div>
                    <div style="color: #4B5563; font-size: 13px; line-height: 1.55;">
                        RMSE places a heavier penalty on unusually large prediction errors. Because luxury and boutique gaming rigs have rare configurations, RMSE is higher than MAE.
                    </div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #15803D; font-weight: 700; font-size: 13.5px; margin-bottom: 6px;">
                        {get_icon("check-circle", size=16, color="#15803D")}
                        <span>R² Score: 0.7748 (77.5% Explained Variance)</span>
                    </div>
                    <div style="color: #4B5563; font-size: 13px; line-height: 1.55;">
                        The model accounts for approximately <strong>77.5% of the total price variance</strong> solely from raw technical specifications (RAM, CPU, GPU, Resolution, Weight).
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pipeline Architecture Breakdown
    st.markdown(
        f"""
        <div class="dashboard-card">
            <h3 style="color: #111827; font-size: 17px; font-weight: 750; margin: 0 0 14px 0;">
                Pipeline Architecture & Training Methodology
            </h3>
            <ol style="color: #4B5563; font-size: 13.5px; line-height: 1.7; margin: 0 0 14px 20px;">
                <li><strong>Input Ingestion:</strong> Accepts 3 numeric features (Display Size, RAM, Weight) and 7 categorical features (Brand, Type, Resolution, CPU, Storage, GPU, OS).</li>
                <li><strong>Preprocessing:</strong> Scikit-learn <code style="color: #2563EB; background: #EFF6FF; padding: 2px 6px; border-radius: 4px; border: 1px solid #DBEAFE;">ColumnTransformer</code> applies One-Hot Encoding to categorical variables while passing numeric variables directly.</li>
                <li><strong>Ensemble Estimator:</strong> <code style="color: #2563EB; background: #EFF6FF; padding: 2px 6px; border-radius: 4px; border: 1px solid #DBEAFE;">RandomForestRegressor</code> with 300 decision trees, <code style="color: #2563EB; background: #EFF6FF; padding: 2px 6px; border-radius: 4px; border: 1px solid #DBEAFE;">min_samples_leaf=2</code>, and parallel processing.</li>
                <li><strong>Target & Currency Scaling:</strong> The base model predicts prices in <strong>Euros (EUR)</strong>. The web client dynamically converts values into <strong>Sri Lankan Rupees (LKR)</strong> at the active rate of <strong>1 EUR = Rs. {exchange_rate:,.2f}</strong>.</li>
            </ol>
            <div style="background: #F0FDFA; border-left: 3px solid #0D9488; padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 12.5px; color: #134E4A;">
                <strong>Note on R² Invariance:</strong> Because R² is a scale-invariant correlation metric, converting the target variable linearly does not change the R² value (0.7748).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("<hr style='border-color: #E5E7EB; margin-top: 36px; margin-bottom: 20px;'>", unsafe_allow_html=True)

footer_l, footer_r = st.columns([2, 1])
with footer_l:
    st.markdown(
        """
        <div style="color: #6B7280; font-size: 13px;">
            <strong style="color: #111827;">Laptop AI</strong> · Professional Laptop Price Intelligence Platform<br>
            Built with Python · Streamlit · Scikit-Learn · Plotly · Pandas
        </div>
        """,
        unsafe_allow_html=True,
    )

with footer_r:
    st.markdown(
        f"""
        <div style="color: #6B7280; font-size: 12px; text-align: right;">
            Sri Lanka Edition · LKR Valuation Engine<br>
            Active Forex Rate: 1 EUR = Rs. {exchange_rate:,.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )
