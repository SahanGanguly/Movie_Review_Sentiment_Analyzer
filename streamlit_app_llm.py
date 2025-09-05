import streamlit as st
from sentiment_llm import SentimentAnalyzer  # import your class

# Initialize analyzer
analyzer = SentimentAnalyzer()
# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type or paste a movie review and click **Analyze** to see sentiment classification.")

review_input = st.text_area("Enter your review here:", height=120)

if st.button("Analyze") and review_input.strip():
    with st.spinner("⏳ Analyzing..."):
        result = analyzer.analyze_sentiment(review_input.strip())

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
