# Movie_Review_Sentiment_Analyzer
# A. Sentiment_LLM: 
    The central idea of this model is to use Gemini's free tier API (Gemini 2.5 Flash in our case) in order to determine a movie review sentiment. 
# Steps (On VS Code or your preferred IDE terminal)
    i. pip install google-generativeai
    ii.pip install python-dotenv

# 1. Imports & Setup
    os, dotenv: Loads the API key from an .env file securely.
    google.generativeai: Connects to the Gemini model.
    re, json: Used for parsing the AI’s response (ensuring it’s valid JSON).
    logging: Adds logs for debugging and tracking.
    hashlib, lru_cache: Used for caching results so repeated reviews don’t trigger new API calls.

2. Model formation:
   
# class SentimentAnalyzer (main engine):
    __init__ configures the Gemini API key and sets parameters like temperature, model name and neutral threshold. Here I could've used the new Gemini 2.5 pro but the Flash model is simply faster and more adept        for simple classification unlike the pro model which is designed for deep thinking. Now here the temperature is set at very low level in order to form consistent results.
      The neutral threshhold has been used for a specific purpose which has been mentioned in the mini report.
# self.few_shot_examples 
    The few shot example(s) trains the model to show how the input looks like and how the output should look like. 
# analyze_sentiment:
    Analyzes the sentiment of a movie review using the Gemini model. Returns a dictionary with label, confidence, explanation, and evidence phrases. The lru_cache helps in storing previous similar reviews as         cache in local folder so that the program doesn't need to re analyze the same review. This is known as light-weight prompt chaining. It calls Gemini’s generate_content to get a response. Passes the response      to _parse_response for cleaning & validation.
# _construct_prompt:
    It prepares a structured prompt with few-shot examples and tells Gemini to only return JSON, no extra text. Adds the user’s review at the end.
# parse_response
    It extracts JSON safely using regex. Validates keys: label, confidence, explanation, evidence_phrases. Ensures that
    label is one of Positive / Negative / Neutral. Confidence is between 0 and 1. Evidence phrases is a list. If the response is invalid then the model defaults to Neutral with confidence 0.5.
# if __name__ == "__main__":
    This is the main execution block where the user enters reviews and each review is sent to analyzer and thereafter it prints results with coloured emojis.

# B.Streamlit_app.py
    Wraps the existing sentiment analyzer around a web based UI.
# Step: streamlit run streamlit_app.py
# 1. Imports and Setup:
     streamlit for developing the web UI
     SentimentAnalyzer from the existing sentiment_llm.py
2.Model Formation:
# Streamlit UI
Sets up page title & layout. Adds a text area for users to input their movie review. Adds an Analyze button to trigger sentiment analysis.

C.batch_eval.py
# Imports and Libraries
    Pandas to load dataset
    Counter: Counts the number of occurences
    Confusion Matrix and Classification Report: Evaluation Metrics for model evaluation

 # Confusion Matrix: 
    Shows how many reviews of each true label were predicted correctly/incorrectly.
 # Classification Report: 
    Provides precision, recall, F1-score for each sentiment class.
