import streamlit as st
import joblib
import re
import os
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for styling and animation
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTitle { color: #1E3D59; font-size: 3rem !important; padding-bottom: 2rem; }
    .prediction-box { padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; background-color: #f0f2f6; text-align: center; }
    .stButton>button { background-color: #1E3D59; color: white; padding: 0.5rem 2rem; width: 100%; }
    .result-text { font-size: 2rem; font-weight: bold; color: #FFFFFF; }
    .fake { background-color: #FF4B4B; animation: fadeIn 1s; }
    .true { background-color: #4CAF50; animation: fadeIn 1s; }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Logo and Title
st.title("🔍 Fake News Detector")

st.info("Our AI-powered tool uses advanced machine learning to help identify potential fake news articles.")

# Function to safely load models
def load_model(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"Model file not found: {file_path}")
            return None
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading model {file_path}: {str(e)}")
        return None

# Load the Passive Aggressive Classifier and TF-IDF Vectorizer
@st.cache_resource
def load_all_models():
    passive_model = load_model('models/pac_model.pkl')
    tfidf_vectorizer = load_model('models/tfidf_vectorizer.pkl')
    return passive_model, tfidf_vectorizer

passive_model, tfidf_vectorizer = load_all_models()

# Check if models loaded successfully
if None in [passive_model, tfidf_vectorizer]:
    st.error("Failed to load the model files. Please check and try again.")
    st.stop()

# Function for text preprocessing
def preprocess_input(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.lower()
    return text

# Input area
st.markdown("### Input Article")
user_input = st.text_area(
    "Paste your article here:",
    height=200,
    placeholder="Enter the news article text here..."
)

# Analyze button
if st.button("Analyze Article 🔎"):
    if user_input:
        with st.spinner('Analyzing the article...'):
            # Preprocess and transform the input text
            processed_input = preprocess_input(user_input)
            input_tfidf = tfidf_vectorizer.transform([processed_input])

            # Make a prediction with the PAC model
            pac_pred = passive_model.predict(input_tfidf)[0]

            # Display result with animation
            time.sleep(1)  # Simulate a slight delay for effect
            st.markdown("### Analysis Result")
            
            # Conditional result display with logo and color
            if pac_pred == 1:
                st.markdown("""
                    <div class="prediction-box true">
                        <p class="result-text">✅ True News</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box fake">
                        <p class="result-text">❌ Fake News</p>
                    </div>
                """, unsafe_allow_html=True)

            # Timestamp
            st.markdown("---")
            st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("⚠️ Please enter a news article to analyze.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This tool is for educational purposes only. Always verify news from multiple reliable sources.</p>
    </div>
""", unsafe_allow_html=True)
