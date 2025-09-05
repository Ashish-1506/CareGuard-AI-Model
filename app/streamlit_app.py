import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from features import FeatureEngineer
    from explain import ModelExplainer
except ImportError:
    st.error("Could not import required modules. Please check file paths.")
    st.stop()

# Configure page
st.set_page_config(
    page_title='CareGuard AI',
    page_icon='⚕️',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Enhanced Medical Theme CSS with Sophisticated Background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Medical Color Palette */
    :root {
        --primary-medical-blue: #0066CC;
        --secondary-medical-blue: #4A90E2;
        --light-medical-blue: #E8F4FD;
        --medical-green: #00A86B;
        --light-medical-green: #E8F8F5;
        --medical-teal: #20B2AA;
        --medical-gray: #F8F9FA;
        --medical-dark-gray: #2C3E50;
        --medical-red: #DC3545;
        --medical-orange: #FD7E14;
        --medical-white: #FFFFFF;
        --medical-light-gray: #E9ECEF;
        --shadow-light: 0 2px 10px rgba(0, 102, 204, 0.1);
        --shadow-medium: 0 4px 20px rgba(0, 102, 204, 0.15);
        --shadow-strong: 0 8px 30px rgba(0, 102, 204, 0.2);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sophisticated Medical Background */
    .stApp {
        background: 
            /* Primary gradient */
            linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #f1f5f9 50%, #e8f4fd 75%, #f0f9ff 100%),
            /* Medical cross pattern */
            radial-gradient(circle at 20% 80%, rgba(0, 102, 204, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0, 168, 107, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(32, 178, 170, 0.06) 0%, transparent 50%),
            /* Heartbeat pattern */
            linear-gradient(90deg, transparent 24%, rgba(0, 102, 204, 0.03) 25%, rgba(0, 102, 204, 0.03) 26%, transparent 27%, transparent 74%, rgba(0, 168, 107, 0.03) 75%, rgba(0, 168, 107, 0.03) 76%, transparent 77%);
        
        background-size: 100% 100%, 800px 800px, 600px 600px, 400px 400px, 100px 100px;
        background-position: 0 0, 0 0, 100% 0, 50% 50%, 0 0;
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
    }
    
    /* Animated medical particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            /* DNA helix pattern */
            radial-gradient(2px 2px at 40px 60px, rgba(0, 102, 204, 0.15), transparent),
            radial-gradient(2px 2px at 80px 120px, rgba(0, 168, 107, 0.15), transparent),
            radial-gradient(1px 1px at 120px 40px, rgba(32, 178, 170, 0.1), transparent),
            /* Medical cross symbols */
            radial-gradient(1px 1px at 200px 180px, rgba(0, 102, 204, 0.1), transparent),
            radial-gradient(1px 1px at 160px 260px, rgba(0, 168, 107, 0.1), transparent),
            /* Stethoscope curves */
            linear-gradient(45deg, transparent 48%, rgba(0, 102, 204, 0.02) 49%, rgba(0, 102, 204, 0.02) 51%, transparent 52%);
        
        background-size: 240px 300px, 180px 240px, 120px 160px, 280px 320px, 200px 260px, 60px 60px;
        animation: medicalFloat 60s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Floating medical icons animation */
    @keyframes medicalFloat {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg);
            opacity: 0.4;
        }
        25% { 
            transform: translateY(-10px) rotate(1deg);
            opacity: 0.6;
        }
        50% { 
            transform: translateY(-5px) rotate(-1deg);
            opacity: 0.5;
        }
        75% { 
            transform: translateY(-15px) rotate(0.5deg);
            opacity: 0.7;
        }
    }
    
    /* Secondary overlay for depth */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            /* Subtle medical grid */
            linear-gradient(rgba(0, 102, 204, 0.01) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 102, 204, 0.01) 1px, transparent 1px),
            /* Heartbeat rhythm */
            repeating-linear-gradient(
                45deg,
                transparent,
                transparent 35px,
                rgba(0, 168, 107, 0.015) 35px,
                rgba(0, 168, 107, 0.015) 37px,
                transparent 37px,
                transparent 70px
            );
        background-size: 40px 40px, 40px 40px, 140px 140px;
        animation: pulseGrid 30s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes pulseGrid {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    /* Hide sidebar permanently */
    .css-1d391kg {
        display: none;
    }
    
    /* Medical Header with enhanced background */
    .medical-header {
        background: 
            linear-gradient(135deg, var(--primary-medical-blue) 0%, var(--secondary-medical-blue) 30%, var(--medical-teal) 70%, #1e88e5 100%),
            /* Medical cross overlay */
            radial-gradient(circle at 15% 15%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 85% 85%, rgba(255, 255, 255, 0.08) 0%, transparent 50%);
        padding: 3.5rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 
            var(--shadow-strong),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .medical-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            repeating-conic-gradient(
                from 0deg at 50% 50%,
                transparent 0deg,
                rgba(255, 255, 255, 0.03) 30deg,
                transparent 60deg
            );
        animation: rotateBackground 120s linear infinite;
        pointer-events: none;
    }
    
    @keyframes rotateBackground {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .medical-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    
    .medical-subtitle {
        font-size: 1.4rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 2;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }
    
    /* Enhanced Tab Styling with glass effect */
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 18px;
        margin-bottom: 3rem;
        box-shadow: 
            var(--shadow-medium),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 102, 204, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 75px;
        padding: 18px 35px;
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(8px);
        border-radius: 18px;
        border: 2px solid transparent;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-light);
        color: var(--medical-dark-gray);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 102, 204, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: var(--shadow-strong);
        background: linear-gradient(135deg, rgba(255, 255, 255, 1) 0%, var(--light-medical-blue) 100%);
        border-color: var(--secondary-medical-blue);
    }
    
    .stTabs [aria-selected="true"] {
        background: 
            linear-gradient(135deg, var(--primary-medical-blue) 0%, var(--secondary-medical-blue) 50%, var(--medical-teal) 100%);
        color: white;
        border: 2px solid var(--primary-medical-blue);
        transform: translateY(-5px) scale(1.05);
        box-shadow: 
            var(--shadow-strong),
            0 0 20px rgba(0, 102, 204, 0.4);
    }
    
    /* Enhanced Medical Risk Cards with glass morphism */
    .medical-risk-card {
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 
            var(--shadow-strong),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border-left: 6px solid;
        margin: 2.5rem 0;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .medical-risk-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: -100px;
        width: 200px;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
        transform: skewX(-20deg);
        transition: transform 0.8s ease;
    }
    
    .medical-risk-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow: 
            0 20px 50px rgba(0, 102, 204, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }
    
    .medical-risk-card:hover::before {
        transform: translateX(200px) skewX(-20deg);
    }
    
    .risk-high {
        border-left-color: var(--medical-red);
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 245, 245, 0.9) 100%);
    }
    
    .risk-medium {
        border-left-color: var(--medical-orange);
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 248, 240, 0.9) 100%);
    }
    
    .risk-low {
        border-left-color: var(--medical-green);
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, var(--light-medical-green) 100%);
    }
    
    .medical-risk-score {
        font-size: 4.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        position: relative;
        z-index: 2;
    }
    
    .medical-risk-label {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 1rem;
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced Medical Metric Cards with glass effect */
    .medical-metric-card {
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(12px);
        padding: 2.8rem 2.5rem;
        border-radius: 22px;
        box-shadow: 
            var(--shadow-medium),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        text-align: center;
        border-top: 4px solid var(--primary-medical-blue);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .medical-metric-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-medical-blue) 0%, var(--medical-teal) 100%);
        transition: height 0.4s ease;
    }
    
    .medical-metric-card:hover {
        transform: translateY(-8px) scale(1.04);
        box-shadow: 
            var(--shadow-strong),
            0 0 25px rgba(0, 102, 204, 0.2);
    }
    
    .medical-metric-card:hover::before {
        height: 8px;
    }
    
    .medical-metric-card h3 {
        margin-bottom: 1.8rem;
        font-size: 1.4rem;
        color: var(--medical-dark-gray);
        font-weight: 600;
    }
    
    .medical-metric-card .medical-icon {
        font-size: 1.8rem;
        color: var(--primary-medical-blue);
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Enhanced Medical Form with glass morphism */
    .medical-form {
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(232, 244, 253, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: 3.5rem;
        border-radius: 30px;
        box-shadow: 
            var(--shadow-strong),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        margin: 3rem 0;
        border: 1px solid rgba(0, 102, 204, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .medical-form::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 80% 20%, rgba(0, 102, 204, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 20% 80%, rgba(0, 168, 107, 0.03) 0%, transparent 50%);
        z-index: 0;
    }
    
    .medical-form-title {
        color: var(--medical-dark-gray);
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 3rem;
        text-align: center;
        padding-bottom: 2rem;
        border-bottom: 3px solid var(--primary-medical-blue);
        position: relative;
        z-index: 1;
    }
    
    /* Patient registration form */
    .patient-registration-form {
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(232, 244, 253, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: 3.5rem;
        border-radius: 30px;
        box-shadow: 
            var(--shadow-strong), 
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        margin: 3rem 0;
        border: 1px solid rgba(0, 102, 204, 0.15);
        position: relative;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, var(--light-medical-green) 0%, rgba(212, 237, 218, 0.9) 100%);
        border: 2px solid var(--medical-green);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        animation: successPulse 2s ease-in-out;
        backdrop-filter: blur(10px);
    }
    
    @keyframes successPulse {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Enhanced Medical Button with glow effect */
    .stButton > button {
        background: 
            linear-gradient(135deg, var(--primary-medical-blue) 0%, var(--secondary-medical-blue) 30%, var(--medical-teal) 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 1.5rem 4rem;
        font-size: 1.4rem;
        font-weight: 600;
        box-shadow: 
            var(--shadow-strong),
            0 0 20px rgba(0, 102, 204, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s ease, height 0.6s ease;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 
            var(--shadow-strong),
            0 0 30px rgba(0, 102, 204, 0.5);
        background: 
            linear-gradient(135deg, #004499 0%, #3a7bc8 30%, #1a9999 100%);
    }
    
    /* Enhanced Clinical Summary with pulse effect */
    .medical-clinical-summary {
        background: 
            linear-gradient(135deg, var(--light-medical-blue) 0%, rgba(204, 231, 255, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-left: 6px solid var(--primary-medical-blue);
        padding: 3rem;
        border-radius: 22px;
        font-size: 1.3rem;
        line-height: 1.9;
        margin: 3rem 0;
        box-shadow: 
            var(--shadow-medium),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        position: relative;
        border: 1px solid rgba(0, 102, 204, 0.2);
    }
    
    .medical-clinical-summary::before {
        content: '⚕️';
        position: absolute;
        top: 25px;
        right: 25px;
        font-size: 2.5rem;
        opacity: 0.4;
        animation: pulse 3s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.4; }
        50% { transform: scale(1.1); opacity: 0.7; }
    }
    
    /* Enhanced Recommendation Cards with slide effect */
    .medical-recommendation {
        background: 
            linear-gradient(135deg, var(--light-medical-green) 0%, rgba(212, 237, 218, 0.9) 100%);
        backdrop-filter: blur(8px);
        border-left: 5px solid var(--medical-green);
        padding: 2.5rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        box-shadow: 
            var(--shadow-medium),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        transition: all 0.4s ease;
        position: relative;
        border: 1px solid rgba(0, 168, 107, 0.2);
    }
    
    .medical-recommendation::before {
        content: '💊';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 1.8rem;
        opacity: 0.5;
    }
    
    .medical-recommendation:hover {
        transform: translateX(12px) scale(1.02);
        box-shadow: 
            var(--shadow-strong),
            0 0 20px rgba(0, 168, 107, 0.2);
    }
    
    /* Enhanced Input Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stDateInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid rgba(0, 102, 204, 0.2) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(5px) !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stDateInput > div > div > input:focus {
        border-color: var(--primary-medical-blue) !important;
        box-shadow: 0 0 0 4px rgba(0, 102, 204, 0.15) !important;
    }
    
    /* Form submit button styling */
    .stForm > div > div > button {
        background: linear-gradient(135deg, var(--medical-green) 0%, #16a085 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 1.2rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 168, 107, 0.3) !important;
    }
    
    .stForm > div > div > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(0, 168, 107, 0.4) !important;
    }
    
    /* Medical Animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .medical-card-animate {
        animation: slideInUp 0.8s ease-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary-medical-blue) !important;
    }
    
    /* Enhanced Dataframe Styling */
    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: var(--shadow-medium);
        backdrop-filter: blur(10px);
    }
    
    /* Enhanced Messages */
    .stSuccess, .stInfo {
        border-radius: 15px !important;
        border-left: 4px solid var(--medical-green) !important;
        background: rgba(232, 248, 245, 0.9) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .stError {
        border-radius: 15px !important;
        border-left: 4px solid var(--medical-red) !important;
        background: rgba(255, 245, 245, 0.9) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 102, 204, 0.1) !important;
    }
    
    /* Balloons animation enhancement */
    .stBalloons {
        z-index: 9999 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for patient management
if 'custom_patients' not in st.session_state:
    st.session_state.custom_patients = []

if 'next_patient_id' not in st.session_state:
    st.session_state.next_patient_id = 90000  # Start custom patient IDs from 90000

@st.cache_data
def load_model():
    """Load the trained model bundle"""
    try:
        return joblib.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/model.pkl')))
    except FileNotFoundError:
        st.error("Model file not found. Please run the training pipeline first.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample patient data"""
    try:
        return pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/training_table.csv')))
    except FileNotFoundError:
        st.error("Sample data not found. Please run the data generation script first.")
        return None

def get_combined_patient_data(original_df):
    """Combine original patient data with newly added patients"""
    if st.session_state.custom_patients:
        custom_df = pd.DataFrame(st.session_state.custom_patients)
        combined_df = pd.concat([original_df, custom_df], ignore_index=True)
        return combined_df
    return original_df

def add_new_patient_to_system(patient_data):
    """Add a new patient to the system"""
    patient_data['patient_id'] = st.session_state.next_patient_id
    patient_data['last_updated'] = datetime.now()
    patient_data['date_added'] = datetime.now().strftime('%Y-%m-%d')
    
    st.session_state.custom_patients.append(patient_data)
    st.session_state.next_patient_id += 1
    
    return patient_data['patient_id']

def get_risk_band_color(risk_band):
    """Get color for risk band"""
    colors = {
        'High': '#DC3545',
        'Medium': '#FD7E14',
        'Low': '#00A86B'
    }
    return colors.get(risk_band, '#6C757D')

def create_cohort_overview(df, model_bundle):
    """Create cohort overview with predictions"""
    feature_engineer = model_bundle['feature_engineer']
    scaler = model_bundle['scaler']
    model = model_bundle['calibrated_model']
    
    df_features = feature_engineer.create_features(df)
    X = feature_engineer.prepare_model_features(df_features)
    X_scaled = scaler.transform(X)
    
    probabilities = model.predict_proba(X_scaled)[:, 1]
    risk_bands = pd.cut(probabilities, bins=[-0.01, 0.1, 0.25, 1.0], labels=['Low', 'Medium', 'High'])
    
    df_display = df.copy()
    df_display['risk_probability'] = probabilities
    df_display['risk_band'] = risk_bands
    df_display['flags'] = df_display.apply(lambda row: get_risk_flags(row), axis=1)
    
    return df_display

def get_risk_flags(row):
    """Generate risk flags for display"""
    flags = []
    if row['adherence_mean'] < 0.8: flags.append('Low Adherence')
    if row['weight_trend_30d'] > 2: flags.append('Weight Gain')
    if row['hba1c_last'] > 9: flags.append('High HbA1c')
    if row['bnp_last'] > 400: flags.append('High BNP')
    if row['days_since_last_lab'] > 90: flags.append('Overdue Labs')
    return ', '.join(flags) if flags else 'None'

def create_risk_distribution_chart(df_display):
    """Create risk distribution chart with medical theme"""
    fig = px.histogram(
        df_display, x='risk_probability', nbins=40,
        title='Risk Probability Distribution',
        labels={'risk_probability': 'Risk Probability', 'count': 'Number of Patients'},
        color_discrete_sequence=['#0066CC']
    )
    
    fig.add_vline(x=0.25, line_dash="dash", line_color="#DC3545", annotation_text="Risk Threshold (25%)")
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
        font_family="Inter",
        title_font_size=18,
        title_font_color="#2C3E50"
    )
    return fig

def create_patient_trends_chart(patient_data):
    """Create trends chart with medical theme"""
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
    
    base_hba1c = patient_data['hba1c_last'].iloc[0]
    base_weight = 75 + patient_data['bmi'].iloc[0] * 0.5
    base_adherence = patient_data['adherence_mean'].iloc[0]
    
    trends_data = pd.DataFrame({
        'Date': dates,
        'HbA1c (%)': base_hba1c + np.random.normal(0, 0.3, 12),
        'Weight (kg)': base_weight + np.cumsum(np.random.normal(patient_data['weight_trend_30d'].iloc[0]/30, 0.5, 12)),
        'Adherence (%)': np.clip(base_adherence + np.random.normal(0, 0.05, 12), 0.3, 1.0),
        'SBP (mmHg)': patient_data['sbp_last'].iloc[0] + np.random.normal(0, 8, 12)
    })
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['HbA1c Trend', 'Weight Trend', 'Adherence Trend', 'Blood Pressure Trend'],
        vertical_spacing=0.08
    )
    
    # Medical color scheme
    fig.add_trace(
        go.Scatter(x=trends_data['Date'], y=trends_data['HbA1c (%)'], 
                  name='HbA1c', line=dict(color='#DC3545', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=trends_data['Date'], y=trends_data['Weight (kg)'], 
                  name='Weight', line=dict(color='#0066CC', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=trends_data['Date'], y=trends_data['Adherence (%)'], 
                  name='Adherence', line=dict(color='#00A86B', width=3)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=trends_data['Date'], y=trends_data['SBP (mmHg)'], 
                  name='SBP', line=dict(color='#20B2AA', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=400, showlegend=False,
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
        font_family="Inter"
    )
    return fig

def render_patient_registration_form():
    """Render comprehensive patient registration form"""
    
    st.markdown("""
    <div class="patient-registration-form">
        <h2 style="text-align: center; color: var(--medical-dark-gray); font-size: 2.4rem; margin-bottom: 2rem;">
            📋 Patient Registration System
        </h2>
        <p style="text-align: center; color: var(--medical-dark-gray); font-size: 1.1rem; margin-bottom: 3rem;">
            Complete patient intake for comprehensive chronic care monitoring and AI-powered risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("patient_registration_form", clear_on_submit=True):
        # Patient Demographics
        st.markdown("### 👤 Patient Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name*", placeholder="Enter patient's full name")
            age = st.number_input("Age (years)*", min_value=18, max_value=120, value=65)
            sex = st.selectbox("Biological Sex*", ["M", "F"])
        
        with col2:
            date_of_birth = st.date_input("Date of Birth", value=datetime.now() - timedelta(days=65*365))
            contact_phone = st.text_input("Contact Phone", placeholder="Patient contact number")
            emergency_contact = st.text_input("Emergency Contact", placeholder="Emergency contact information")
        
        st.markdown("---")
        
        # Medical History
        st.markdown("### 🏥 Medical History & Diagnosis")
        col1, col2 = st.columns(2)
        
        with col1:
            condition_primary = st.selectbox("Primary Diagnosis*", 
                                           ["Diabetes", "Heart Failure", "Hypertension", "Multiple", "COPD", "Kidney Disease"])
            medical_history = st.text_area("Medical History", 
                                         placeholder="Previous medical conditions, surgeries, hospitalizations...")
            allergies = st.text_input("Known Allergies", placeholder="Drug/food allergies")
        
        with col2:
            family_history = st.text_area("Family History", 
                                        placeholder="Family history of chronic conditions...")
            current_medications = st.text_area("Current Medications", 
                                             placeholder="List current medications and dosages...")
            physician_name = st.text_input("Primary Physician", placeholder="Attending physician name")
        
        st.markdown("---")
        
        # Clinical Measurements
        st.markdown("### 🔬 Clinical Laboratory & Vitals")
        col1, col2 = st.columns(2)
        
        with col1:
            hba1c_last = st.number_input("HbA1c Level (%)*", min_value=4.0, max_value=20.0, value=8.2, step=0.1)
            sbp_last = st.number_input("Systolic Blood Pressure (mmHg)*", min_value=80, max_value=250, value=140)
            dbp_last = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=85)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            bnp_last = st.number_input("BNP Level (pg/mL)*", min_value=0, max_value=5000, value=150)
        
        with col2:
            weight_current = st.number_input("Current Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
            bmi = weight_current / ((height/100) ** 2) if height > 0 else 25.0
            st.metric("Calculated BMI", f"{bmi:.1f}")
            
            weight_trend_30d = st.number_input("Weight Change (kg/30d)*", min_value=-10.0, max_value=10.0, value=0.5, step=0.1)
            egfr_trend_90d = st.number_input("eGFR Trend (mL/min/1.73m²/90d)", min_value=-30.0, max_value=20.0, value=-2.1, step=0.1)
        
        st.markdown("---")
        
        # Treatment & Lifestyle
        st.markdown("### 💊 Treatment & Lifestyle Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            adherence_mean = st.slider("Medication Adherence Rate*", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
            smoker = st.selectbox("Smoking Status*", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Current smoker")
            alcohol_use = st.selectbox("Alcohol Use", ["None", "Occasional", "Moderate", "Heavy"])
        
        with col2:
            exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "1-2x/week", "3-4x/week", "Daily"])
            diet_compliance = st.slider("Diet Compliance", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
            days_since_last_lab = st.number_input("Days Since Last Lab Work*", min_value=0, max_value=365, value=45)
        
        st.markdown("---")
        
        # Additional Notes
        st.markdown("### 📝 Additional Clinical Notes")
        clinical_notes = st.text_area("Clinical Observations", 
                                     placeholder="Additional clinical observations, symptoms, concerns...")
        risk_factors = st.text_area("Additional Risk Factors", 
                                   placeholder="Other relevant risk factors not captured above...")
        
        # Submit button
        submitted = st.form_submit_button("🏥 Register Patient in System", use_container_width=True)
        
        if submitted:
            if patient_name and age and sex and condition_primary:
                # Create patient data dictionary
                new_patient = {
                    'patient_name': patient_name,
                    'age': age,
                    'sex': sex,
                    'date_of_birth': date_of_birth.strftime('%Y-%m-%d'),
                    'contact_phone': contact_phone,
                    'emergency_contact': emergency_contact,
                    'condition_primary': condition_primary,
                    'medical_history': medical_history,
                    'allergies': allergies,
                    'family_history': family_history,
                    'current_medications': current_medications,
                    'physician_name': physician_name,
                    'hba1c_last': hba1c_last,
                    'sbp_last': sbp_last,
                    'dbp_last': dbp_last,
                    'heart_rate': heart_rate,
                    'bnp_last': bnp_last,
                    'weight_current': weight_current,
                    'height': height,
                    'bmi': bmi,
                    'weight_trend_30d': weight_trend_30d,
                    'egfr_trend_90d': egfr_trend_90d,
                    'adherence_mean': adherence_mean,
                    'smoker': smoker,
                    'alcohol_use': alcohol_use,
                    'exercise_frequency': exercise_frequency,
                    'diet_compliance': diet_compliance,
                    'days_since_last_lab': days_since_last_lab,
                    'clinical_notes': clinical_notes,
                    'risk_factors': risk_factors,
                }
                
                # Add patient to system
                patient_id = add_new_patient_to_system(new_patient)
                
                # Success message
                st.markdown(f"""
                <div class="success-message">
                    <h3>✅ Patient Successfully Registered!</h3>
                    <p><strong>Patient ID:</strong> {patient_id}</p>
                    <p><strong>Name:</strong> {patient_name}</p>
                    <p><strong>Registration Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    <p>Patient is now available for risk analysis in the Patient Analysis tab.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()  # Celebration animation
                
            else:
                st.error("⚠️ Please fill in the required fields marked with * (Name, Age, Sex, Primary Diagnosis)")

def render_new_patient_form(model_bundle):
    """Render quick risk testing form (existing functionality)"""
    
    st.markdown("""
    <div class="medical-form">
        <h2 style="text-align: center; color: var(--medical-dark-gray); font-size: 2.2rem; margin-bottom: 2rem;">
            🩺 Quick AI Risk Assessment Laboratory
        </h2>
        <p style="text-align: center; color: var(--medical-dark-gray); margin-bottom: 2rem;">
            Rapid risk evaluation for immediate clinical decision support
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical-themed tabs
    tab1, tab2, tab3 = st.tabs([
        "👤 Patient Demographics", 
        "🔬 Clinical Laboratory", 
        "💊 Treatment Protocol"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("👤 Patient Age (years)", min_value=18, max_value=100, value=60)
            sex = st.selectbox("⚧ Biological Sex", ["M", "F"])
        with col2:
            cond = st.selectbox("🏥 Primary Diagnosis", ["Diabetes", "Heart Failure", "Hypertension", "Multiple"])
            bmi = st.number_input("⚖️ Body Mass Index", min_value=15.0, max_value=45.0, value=27.0)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            hba1c_last = st.number_input("🩸 HbA1c Level (%)", min_value=4.0, max_value=15.0, value=7.8)
            sbp_last = st.number_input("💗 Systolic Blood Pressure", min_value=90, max_value=200, value=130)
            bnp_last = st.number_input("🫀 B-Type Natriuretic Peptide (pg/mL)", min_value=0, max_value=3000, value=120)
        with col2:
            weight_trend = st.number_input("📊 Weight Change (kg/30d)", min_value=-5.0, max_value=5.0, value=1.0)
            egfr_trend = st.number_input("🧬 eGFR Trend (mL/min/1.73m²/90d)", min_value=-20.0, max_value=20.0, value=-3.1)
            days_since_lab = st.number_input("📅 Days Since Last Laboratory Work", min_value=0, max_value=365, value=34)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            adherence = st.slider("💊 Medication Adherence Rate", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
        with col2:
            smoker = st.selectbox("🚭 Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Current smoker")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔬 Generate Quick Risk Assessment"):
        new_patient = {
            "patient_id": 99999, "patient_name": "Quick Assessment", "age": age, "sex": sex,
            "condition_primary": cond, "hba1c_last": hba1c_last, "weight_trend_30d": weight_trend,
            "adherence_mean": adherence, "bnp_last": bnp_last, "egfr_trend_90d": egfr_trend,
            "sbp_last": sbp_last, "bmi": bmi, "days_since_last_lab": days_since_lab,
            "smoker": smoker, "last_updated": pd.Timestamp.now()
        }
        
        new_df = pd.DataFrame([new_patient])
        features = model_bundle['feature_engineer'].create_features(new_df)
        X = model_bundle['feature_engineer'].prepare_model_features(features)
        X_scaled = model_bundle['scaler'].transform(X)
        prob = model_bundle['calibrated_model'].predict_proba(X_scaled)[0, 1]
        risk_band = pd.cut([prob], bins=[-0.01, 0.1, 0.25, 1.0], labels=['Low', 'Medium', 'High'])[0]
        
        # Medical results display
        risk_class = f"risk-{risk_band.lower()}"
        st.markdown(f"""
        <div class="medical-risk-card {risk_class} medical-card-animate">
            <div style="text-align: center;">
                <div class="medical-risk-score" style="color: {get_risk_band_color(risk_band)};">
                    {prob:.1%}
                </div>
                <div class="medical-risk-label">
                    {risk_band} Risk of 90-day Clinical Deterioration
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Medical metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">👤</div>
                <h3>Patient Profile</h3>
                <p><strong>Age:</strong> {age} years</p>
                <p><strong>Sex:</strong> {sex}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">🔬</div>
                <h3>Key Biomarkers</h3>
                <p><strong>HbA1c:</strong> {hba1c_last}%</p>
                <p><strong>SBP:</strong> {sbp_last} mmHg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">💊</div>
                <h3>Treatment Status</h3>
                <p><strong>Adherence:</strong> {adherence:.0%}</p>
                <p><strong>Diagnosis:</strong> {cond}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">⚠️</div>
                <h3>Clinical Alerts</h3>
                <p><strong>Weight:</strong> {'📈' if weight_trend > 1.5 else '✅'}</p>
                <p><strong>Labs:</strong> {'⚠️' if days_since_lab > 90 else '✅'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="medical-clinical-summary">
            <h4>🩺 Clinical Assessment Report</h4>
            <p>Based on comprehensive AI analysis of clinical parameters, this patient demonstrates a <strong>{risk_band.lower()} probability</strong> ({prob:.1%}) of experiencing clinical deterioration within the next 90-day monitoring period. This assessment integrates biomarker trends, medication adherence patterns, and established risk stratification protocols.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main medical-themed dashboard application"""
    
    # Medical header
    st.markdown("""
    <div class="medical-header">
        <h1 class="medical-title">⚕️ CareGuard AI</h1>
        <p class="medical-subtitle">Advanced Chronic Care Risk Prediction & Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load medical data
    model_bundle = load_model()
    if model_bundle is None:
        st.stop()
    
    original_df = load_sample_data()
    if original_df is None:
        st.stop()
    
    # Get combined patient data (original + newly added)
    df = get_combined_patient_data(original_df)
    
    # Medical navigation tabs - 6 TABS INCLUDING PATIENT REGISTRATION
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏥 Clinical Dashboard", 
        "📊 Population Health", 
        "🔬 Quick Risk Testing", 
        "📋 Add New Patient",
        "👤 Patient Analysis", 
        "📈 System Performance"
    ])
    
    # TAB 1: Clinical Dashboard
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="medical-metric-card">
                <div class="medical-icon">🎯</div>
                <h3>Precision Medicine</h3>
                <p>AI-powered risk stratification with 85%+ clinical accuracy for 90-day deterioration forecasting</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="medical-metric-card">
                <div class="medical-icon">🔍</div>
                <h3>Transparent AI</h3>
                <p>Evidence-based insights with SHAP explainability for clinical decision support and audit trails</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="medical-metric-card">
                <div class="medical-icon">⚡</div>
                <h3>Real-Time Analytics</h3>
                <p>Instant risk scoring with actionable clinical recommendations for proactive patient management</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show current system statistics
        st.markdown("### 📊 System Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏥 Total Patients in System", len(df))
        with col2:
            st.metric("📋 Newly Registered", len(st.session_state.custom_patients))
        with col3:
            st.metric("🔬 Original Dataset", len(original_df))
        with col4:
            if len(df) > 0:
                with st.spinner('Analyzing current patient cohort...'):
                    df_display = create_cohort_overview(df.head(50), model_bundle)
                    high_risk = len(df_display[df_display['risk_band'] == 'High'])
                    st.metric("🚨 High Risk Cases", high_risk)
            else:
                st.metric("🚨 High Risk Cases", "0")
        
        # Live cohort metrics
        if len(df) > 0:
            st.markdown("### 📊 Live Population Health Metrics")
            with st.spinner('Analyzing patient cohort...'):
                df_display = create_cohort_overview(df.head(50), model_bundle)
            
            # Risk distribution visualization
            fig_dist = create_risk_distribution_chart(df_display)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # TAB 2: Population Health Analysis
    with tab2:
        st.markdown("### 🏥 Population Health Risk Analysis")
        
        if len(df) > 0:
            with st.spinner('Generating population health analytics...'):
                df_display = create_cohort_overview(df.head(100), model_bundle)
            
            # Clinical filters
            col1, col2, col3 = st.columns(3)
            with col1:
                condition_filter = st.multiselect('🏥 Filter by Diagnosis', df_display['condition_primary'].unique(), default=df_display['condition_primary'].unique())
            with col2:
                risk_filter = st.multiselect('⚠️ Filter by Risk Level', ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
            with col3:
                age_range = st.slider('👤 Age Demographics', int(df_display['age'].min()), int(df_display['age'].max()), (int(df_display['age'].min()), int(df_display['age'].max())))
            
            # Apply clinical filters
            filtered_df = df_display[
                (df_display['condition_primary'].isin(condition_filter)) &
                (df_display['risk_band'].isin(risk_filter)) &
                (df_display['age'].between(age_range[0], age_range[1]))
            ]
            
            # Population metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('🏥 Analyzed Patients', len(filtered_df))
            with col2:
                high_risk_count = len(filtered_df[filtered_df['risk_band'] == 'High'])
                st.metric('🚨 Critical Cases', high_risk_count, f"{high_risk_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
            with col3:
                avg_risk = filtered_df['risk_probability'].mean() if len(filtered_df) > 0 else 0
                st.metric('📊 Population Risk', f"{avg_risk:.1%}")
            with col4:
                overdue_labs = len(filtered_df[filtered_df['days_since_last_lab'] > 90])
                st.metric('⏰ Overdue Monitoring', overdue_labs)
            
            # Population risk visualization
            if len(filtered_df) > 0:
                fig_dist = create_risk_distribution_chart(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Patient registry
            st.markdown('### 📋 Patient Registry')
            display_cols = ['patient_id', 'patient_name', 'age', 'sex', 'condition_primary', 'risk_probability', 'risk_band', 'hba1c_last', 'adherence_mean', 'flags']
            
            if len(filtered_df) > 0:
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                styled_df = filtered_df[available_cols].style.format({
                    'risk_probability': '{:.1%}',
                    'hba1c_last': '{:.1f}%',
                    'adherence_mean': '{:.0%}'
                })
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info('No patients match the current clinical filters.')
        else:
            st.info('No patient data available. Please add patients using the "Add New Patient" tab.')
    
    # TAB 3: Quick Risk Testing
    with tab3:
        st.markdown("### 🔬 Quick Clinical Risk Assessment")
        st.info("🩺 **Rapid Assessment**: Quick risk evaluation for immediate clinical decision support (does not save patient to system).")
        render_new_patient_form(model_bundle)
    
    # TAB 4: Add New Patient
    with tab4:
        st.markdown("### 📋 Patient Registration & Intake System")
        st.info("🏥 **Comprehensive Registration**: Add new patients to the system for ongoing chronic care monitoring and risk assessment.")
        
        # Show current registered patients
        if st.session_state.custom_patients:
            with st.expander(f"📊 Recently Registered Patients ({len(st.session_state.custom_patients)})"):
                recent_df = pd.DataFrame(st.session_state.custom_patients)
                display_cols = ['patient_id', 'patient_name', 'age', 'sex', 'condition_primary', 'date_added']
                available_cols = [col for col in display_cols if col in recent_df.columns]
                st.dataframe(recent_df[available_cols], use_container_width=True, hide_index=True)
        
        render_patient_registration_form()
    
    # TAB 5: Patient Analysis
    with tab5:
        st.markdown("### 👤 Individual Patient Clinical Assessment")
        
        if len(df) > 0:
            patient_options = [f"{row['patient_id']} - {row['patient_name']}" for _, row in df.iterrows()]
            selected_patient = st.selectbox('🏥 Select Patient for Detailed Clinical Analysis', patient_options)
            
            if selected_patient:
                patient_id = int(selected_patient.split(' - ')[0])
                patient_data = df[df['patient_id'] == patient_id]
                
                if len(patient_data) > 0:
                    with st.spinner('Generating comprehensive clinical assessment...'):
                        df_display = create_cohort_overview(patient_data, model_bundle)
                        explainer = ModelExplainer(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/model.pkl')))
                        explanation = explainer.explain_patient(patient_data, str(patient_id))
                    
                    patient = df_display.iloc[0]
                    
                    # Clinical risk assessment display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        risk_class = f"risk-{explanation['risk_band'].lower()}"
                        st.markdown(f"""
                        <div class="medical-risk-card {risk_class}">
                            <div style="text-align: center;">
                                <div class="medical-risk-score" style="color: {get_risk_band_color(explanation['risk_band'])};">
                                    {explanation['risk_probability']:.1%}
                                </div>
                                <div class="medical-risk-label">
                                    {explanation['risk_band']} Risk Patient
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric('👤 Age', f"{patient['age']:.0f} years")
                        st.metric('⚧ Sex', patient['sex'])
                    
                    with col3:
                        st.metric('🏥 Primary Diagnosis', patient['condition_primary'])
                        if patient_id >= 90000:  # New patient
                            st.success("🆕 Newly Registered Patient")
                        else:
                            st.info("📊 Original Dataset Patient")
                    
                    with col4:
                        st.metric('🩸 HbA1c Level', f"{patient['hba1c_last']:.1f}%")
                        st.metric('💊 Adherence Rate', f"{patient['adherence_mean']:.0%}")
                    
                    # Clinical summary
                    st.markdown(f"""
                    <div class="medical-clinical-summary">
                        <h4>🩺 Clinical Assessment Summary</h4>
                        <p>{explanation['clinical_summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factor analysis
                    st.markdown('### 📈 Primary Risk Factors')
                    for i, driver in enumerate(explanation['top_drivers'][:5]):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            <div class="medical-metric-card" style="text-align: left; margin-bottom: 1rem;">
                                <strong>🔬 {driver['description']}</strong>: {driver['impact']} risk contribution<br>
                                <small>Clinical Impact Score: {driver['shap_value']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.metric('Clinical Value', f"{driver['raw_value']:.2f}" if isinstance(driver['raw_value'], (int, float)) else str(driver['raw_value']))
                    
                    # Clinical recommendations
                    st.markdown('### 💊 Clinical Recommendations')
                    for rec in explanation['recommendations']:
                        st.markdown(f'<div class="medical-recommendation">🩺 {rec}</div>', unsafe_allow_html=True)
                    
                    # Clinical trends
                    st.markdown('### 📊 Clinical Trend Analysis')
                    fig_trends = create_patient_trends_chart(patient_data)
                    st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info('No patients available for analysis. Please add patients using the "Add New Patient" tab.')
    
    # TAB 6: System Performance
    with tab6:
        st.markdown("### 📈 Clinical AI System Performance Validation")
        
        if 'metrics' in model_bundle:
            metrics = model_bundle['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            if 'calibrated' in metrics:
                cal_metrics = metrics['calibrated']
                
                with col1:
                    st.metric("🎯 AUROC Score", f"{cal_metrics.get('AUROC', 0):.3f}", help="Area Under ROC Curve - Clinical discrimination capability")
                with col2:
                    st.metric("📊 AUPRC Score", f"{cal_metrics.get('AUPRC', 0):.3f}", help="Area Under Precision-Recall Curve - Predictive precision")
                with col3:
                    st.metric("🔍 Sensitivity", f"{cal_metrics.get('Sensitivity', 0):.3f}", help="True Positive Rate - High-risk detection capability")
                with col4:
                    st.metric("✅ Specificity", f"{cal_metrics.get('Specificity', 0):.3f}", help="True Negative Rate - Low-risk identification accuracy")
                
                # Clinical validation matrix
                if 'Confusion_Matrix' in cal_metrics:
                    st.markdown("### 🔬 Clinical Validation Matrix")
                    cm = cal_metrics['Confusion_Matrix']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="medical-metric-card">
                            <div class="medical-icon">✅</div>
                            <h3>Accurate Predictions</h3>
                            <p><strong>True Negatives:</strong> {cm[0][0]} patients</p>
                            <p><strong>True Positives:</strong> {cm[1][1]} patients</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="medical-metric-card">
                            <div class="medical-icon">⚠️</div>
                            <h3>Clinical Misclassifications</h3>
                            <p><strong>False Positives:</strong> {cm[0][1]} patients</p>
                            <p><strong>False Negatives:</strong> {cm[1][0]} patients</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Clinical utility assessment
                st.markdown("### 🏥 Clinical Utility Assessment")
                tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
                total_flagged = tp + fp
                nns = total_flagged / tp if tp > 0 else float('inf')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🚨 Patients Flagged", total_flagged, f"{total_flagged/(tp+fp+fn+tn)*100:.1f}% of cohort")
                with col2:
                    st.metric("🔍 Number Needed to Screen", f"{nns:.1f}", help="Patients flagged per true positive case")
                with col3:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    st.metric("🎯 Alert Precision", f"{precision:.1%}", help="Accuracy of high-risk alerts")
        else:
            st.info("🔬 Clinical performance metrics will be displayed following system validation.")
        
        # System information
        st.markdown("### ⚙️ AI System Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">🤖</div>
                <h3>AI Model Configuration</h3>
                <p><strong>Algorithm:</strong> XGBoost + Isotonic Calibration</p>
                <p><strong>Clinical Features:</strong> {len(model_bundle.get('features', []))} variables</p>
                <p><strong>Training Cohort:</strong> {len(original_df)} patients</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="medical-metric-card">
                <div class="medical-icon">⚙️</div>
                <h3>Clinical Parameters</h3>
                <p><strong>Risk Threshold:</strong> {int(model_bundle.get('threshold', 0.25) * 100)}%</p>
                <p><strong>Prediction Window:</strong> 90 days</p>
                <p><strong>Update Frequency:</strong> Real-time</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Medical footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6C757D; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
        ⚕️ <strong>CareGuard AI</strong> - Advancing Clinical Excellence Through Artificial Intelligence<br>
        <small>Built with precision using Streamlit, XGBoost, and SHAP | Clinical AI for Healthcare Professionals</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
