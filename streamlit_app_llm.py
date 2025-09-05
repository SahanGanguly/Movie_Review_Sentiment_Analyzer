import streamlit as st
import os
import google.generativeai as genai
import json
import hashlib
import re
from dotenv import load_dotenv
from functools import lru_cache

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

# Initialize Gemini model
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Cache analyzer to avoid repeated calls for same input
@lru_cache(maxsize=1000)
def analyze_sentiment(review_text: str) -> dict:
    prompt = f"""
You are a sentiment analyzer for movie reviews. Classify sentiment as Positive, Negative, or Neutral.
Return ONLY a JSON object with keys: label, confidence, explanation, evidence_phrases.

Review: {review_text}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 500},
        )
        raw_text = response.text.strip()
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            result = json.loads(json_match.group())
            # Validate
            result["label"] = result.get("label", "Neutral")
            result["confidence"] = float(result.get("confidence", 0.5))
            result["explanation"] = result.get("explanation", "")
            result["evidence_phrases"] = result.get("evidence_phrases", [])
            return result
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "explanation": str(e), "evidence_phrases": []}

    return {"label": "Neutral", "confidence": 0.5, "explanation": "Failed to parse response.", "evidence_phrases": []}

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type or paste a movie review and click **Analyze** to see sentiment classification.")

review_input = st.text_area("Enter your review here:", height=120)

if st.button("Analyze") and review_input.strip():
    with st.spinner("⏳ Analyzing..."):
        result = analyze_sentiment(review_input.strip())

    # Map label to colors
    sentiment_color = {
        "Positive": "#2ecc71",  # green
        "Negative": "#e74c3c",  # red
        "Neutral": "#f1c40f",   # yellow
        "Error": "#95a5a6"      # gray
    }
    color = sentiment_color.get(result["label"], "#95a5a6")

    # Display sentiment badge
    st.markdown(
        f"<h2 style='color:{color}'>Sentiment: {result['label']}</h2>",
        unsafe_allow_html=True
    )

    # Display confidence as progress bar
    st.progress(min(max(result["confidence"], 0.0), 1.0))

    # Display explanation
    st.subheader("Explanation")
    st.write(result.get("explanation", ""))

    # Display evidence phrases
    evidence_phrases = result.get("evidence_phrases", [])
    if evidence_phrases:
        st.subheader("Evidence Phrases")
        st.markdown(", ".join([f"**{p}**" for p in evidence_phrases]))

    # Optional: highlight phrases in the review
    highlighted_text = review_input
    for phrase in evidence_phrases:
        highlighted_text = highlighted_text.replace(phrase, f"<mark>{phrase}</mark>")
    st.subheader("Highlighted Review")
    st.markdown(f"<p>{highlighted_text}</p>", unsafe_allow_html=True)

    # Display JSON output
    st.subheader("Full JSON Output")
    st.json(result, expanded=True)
