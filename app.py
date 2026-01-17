import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .real-news {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .fake-news {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        color: #7F1D1D;
    }
    .confidence-meter {
        height: 20px;
        background-color: #E5E7EB;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses machine learning to detect whether a news article is **REAL** or **FAKE**.
Enter a news article in the text box below and click **Analyze** to check its authenticity.
""")

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('fake_news_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run save_model.py first.")
        return None, None

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and stopword removal
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Load model and vectorizer
model, vectorizer = load_model()

# Sidebar for additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This Fake News Detector uses:
    - **TF-IDF Vectorization** for text processing
    - **Logistic Regression** for classification
    - Trained on a dataset of labeled news articles
    
    ### How it works:
    1. Enter a news article in the text box
    2. The app preprocesses the text
    3. TF-IDF features are extracted
    4. Model predicts if it's REAL or FAKE
    5. Confidence score is displayed
    """)
    
    st.header("📊 Model Info")
    if model is not None:
        st.write(f"**Model Type:** Logistic Regression")
        st.write(f"**Classes:** REAL (0), FAKE (1)")
    
    st.header("⚠️ Disclaimer")
    st.write("""
    This tool is for educational purposes only.
    Always verify information from multiple reliable sources.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Text input area
    st.subheader("📝 Enter News Article")
    news_text = st.text_area(
        "Paste the news article here:",
        height=300,
        placeholder="Enter the full news article text here..."
    )
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    
    if analyze_button:
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        elif model is None or vectorizer is None:
            st.error("Model not loaded. Please check if model files exist.")
        else:
            with st.spinner("Analyzing the article..."):
                # Preprocess the text
                processed_text = preprocess_text(news_text)
                
                # Transform using the vectorizer
                text_features = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_features)[0]
                prediction_proba = model.predict_proba(text_features)[0]
                
                # Calculate confidence
                confidence = max(prediction_proba) * 100
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Result box
                if prediction == 0:  # REAL
                    st.markdown(
                        f'<div class="result-box real-news">✅ This news appears to be REAL</div>',
                        unsafe_allow_html=True
                    )
                else:  # FAKE
                    st.markdown(
                        f'<div class="result-box fake-news">🚨 This news appears to be FAKE</div>',
                        unsafe_allow_html=True
                    )
                
                # Confidence meter
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
                fill_color = "#10B981" if prediction == 0 else "#EF4444"
                st.markdown(
                    f'<div class="confidence-fill" style="width: {confidence}%; background-color: {fill_color};"></div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed probabilities
                col_proba1, col_proba2 = st.columns(2)
                with col_proba1:
                    st.metric("Probability of being REAL", f"{prediction_proba[0]*100:.2f}%")
                with col_proba2:
                    st.metric("Probability of being FAKE", f"{prediction_proba[1]*100:.2f}%")

with col2:
    # Examples section
    st.subheader("📋 Example Articles")
    
    example_tab1, example_tab2 = st.tabs(["REAL Example", "FAKE Example"])
    
    with example_tab1:
        st.write("**Sample REAL news:**")
        st.code("""Scientists have discovered a new species 
of deep-sea fish that glows in the dark. 
The research was published in the 
Journal of Marine Biology after 
extensive peer review.""", language="text")
        if st.button("Use REAL Example", key="real_example"):
            st.session_state.news_text = """Scientists have discovered a new species of deep-sea fish that glows in the dark. The research was published in the Journal of Marine Biology after extensive peer review. The team spent three years studying marine life in the Pacific Ocean."""
    
    with example_tab2:
        st.write("**Sample FAKE news:**")
        st.code("""BREAKING: Government announces 
aliens are real and living among us. 
Official statement expected tonight. 
Share this immediately!""", language="text")
        if st.button("Use FAKE Example", key="fake_example"):
            st.session_state.news_text = """BREAKING: Government announces aliens are real and living among us. Official statement expected tonight. Share this immediately! Sources say they've been here for decades."""

# Handle example text
if 'news_text' in st.session_state:
    news_text = st.session_state.news_text

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "Fake News Detector | Made with Streamlit | Educational Purpose"
    "</div>",
    unsafe_allow_html=True
)

