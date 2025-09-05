import os
import google.generativeai as genai
import re
import json
from dotenv import load_dotenv
import logging
import hashlib
from functools import lru_cache
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = st.secrets.get["GEMINI_API_KEY"]

class SentimentAnalyzer:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.3, neutral_threshold=0.5):
        """
        Initialize the Gemini model for sentiment analysis.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "❌ GEMINI_API_KEY environment variable not set. "
                "Run: $env:GEMINI_API_KEY='your_key_here'"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.neutral_threshold = neutral_threshold

        # Few-shot examples for prompt
        self.few_shot_examples = [
            {
                "review": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
                "label": "Positive",
                "explanation": "The review contains strongly positive words like 'fantastic', 'superb', and 'engaged'.",
                "evidence_phrases": ["absolutely fantastic", "acting was superb", "plot kept me engaged"],
            }
        ]

    @lru_cache(maxsize=1000)
    def analyze_sentiment(self, review_text: str) -> dict:
        """
        Analyze the sentiment of a movie review using the Gemini model.
        Returns a dictionary with label, confidence, explanation, and evidence phrases.
        """
        if not review_text or not isinstance(review_text, str):
            return {
                "label": "Error",
                "confidence": 0.0,
                "explanation": "Invalid input: Review text must be a non-empty string",
                "evidence_phrases": [],
            }

        # Create a hash of the review for caching
        review_hash = hashlib.md5(review_text.encode()).hexdigest()
        logger.info(f"Analyzing review with hash: {review_hash}")

        prompt = self._construct_prompt(review_text)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 500,
                },
            )

            if not response or not response.text:
                raise ValueError("Empty response from model")

            response_text = response.text
            return self._parse_response(response_text)

        except Exception as e:
            # Log the error for debugging
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return {
                "label": "Error",
                "confidence": 0.0,
                "explanation": f"Analysis failed: {str(e)}",
                "evidence_phrases": [],
            }

    def _construct_prompt(self, review_text: str) -> str:
        """
        Build the prompt with few-shot examples and clearer instructions.
        """
        examples_text = ""
        for example in self.few_shot_examples:
            examples_text += f"Review: {example['review']}\n"
            examples_text += f"Label: {example['label']}\n"
            examples_text += f"Explanation: {example['explanation']}\n"
            examples_text += f"Evidence Phrases: {json.dumps(example['evidence_phrases'])}\n\n"

        prompt = f"""
You are a highly-tuned sentiment analysis system specializing in movie reviews. 
You must be aware that movie reviews can contain mixed sentiments, sarcasm, and nuanced opinions.
Your task is to classify the sentiment of the given review as Positive, Negative, or Neutral.

IMPORTANT: You MUST respond with ONLY a valid JSON object with the following structure:
{{
  "label": "Positive | Negative | Neutral",
  "confidence": 0.00,  # A float between 0 and 1
  "explanation": "Short reason grounded in the text",
  "evidence_phrases": ["phrase1", "phrase2"]  # List of phrases that support your conclusion
}}

Do not include any other text before or after the JSON object.

Here are some examples:

{examples_text}

Now analyze this review:

Review: {review_text}

JSON Response:
"""
        return prompt

    def _parse_response(self, response_text: str) -> dict:
        """
        Extract JSON from model output safely with better validation.
        """
        cleaned_text = response_text.strip()
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)

        if json_match:
            try:
                json_str = json_match.group()
                result = json.loads(json_str)

                required_keys = ["label", "confidence", "explanation", "evidence_phrases"]
                if not all(key in result for key in required_keys):
                    raise ValueError("Missing required keys in response")

                # Validate label
                if result["label"] not in ["Positive", "Negative", "Neutral"]:
                    raise ValueError(f"Invalid label: {result['label']}")

                # Validate and convert confidence
                try:
                    confidence = float(result["confidence"])
                    if not 0 <= confidence <= 1:
                        confidence = max(0, min(1, confidence))  # clamp to [0, 1]
                    result["confidence"] = confidence
                except (ValueError, TypeError):
                    result["confidence"] = 0.5

                # 👉 Override rule: low confidence = Neutral
                if result["confidence"] < self.neutral_threshold:
                    result["label"] = "Neutral"

                # Ensure evidence_phrases is a list
                if not isinstance(result["evidence_phrases"], list):
                    result["evidence_phrases"] = []

                return result

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text}")

        # Fallback
        return {
            "label": "Neutral",
            "confidence": 0.5,
            "explanation": f"Could not parse sentiment analysis response. Raw: {response_text[:100]}...",
            "evidence_phrases": [],
        }


if __name__ == "__main__":
    try:
        analyzer = SentimentAnalyzer(neutral_threshold=0.5)  # 👈 easy to tune here
        print("🎬 Movie Review Sentiment Analyzer")
        print("=" * 60)
        previous_explanation = None  # Store previous explanation for chaining
        while True:
            review = input("\n📝 Enter a movie review (or type 'quit' to exit): ").strip()
            if review.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not review:
                print("⚠️  Please enter a review.")
                continue

            chained_review = review
            if previous_explanation:
                chained_review += f"\n\nPrevious explanation for context: {previous_explanation}"

            print("⏳ Analyzing...")
            result = analyzer.analyze_sentiment(chained_review)

            if result["label"] == "Error":
                print("❌ Analysis failed. Please try again.")
                continue

            sentiment_color = {
                "Positive": "🟢",
                "Negative": "🔴",
                "Neutral": "🟡"
            }.get(result["label"], "⚪")

            print(f"\n{sentiment_color} SENTIMENT: {result['label']}")
            print(f"📊 CONFIDENCE: {result['confidence']:.2f}")
            print(f"💡 EXPLANATION: {result['explanation']}")
            print(f"🔍 EVIDENCE: {', '.join(result['evidence_phrases'])}")
            print("=" * 50)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")



