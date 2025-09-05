# Prompt 
    You are a highly-tuned sentiment analysis system specializing in movie reviews. You must be aware that movie reviews can contain mixed sentiments, sarcasm, and nuanced opinions.
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
 # Prompting Logic:
    The keyword 'highly-tuned sentiment analysis system' ensures that the AI model knows exactly how to behave and produces its absolute best result given the constraints. 
    Now the main issue is handling neutral sentiments as they're in the grey zone and LLMs inherently have the tendency to classify sentiments into positive or negative and reviews more than so often contain
    mixed opinions, sarcasm etc. The output format has been given so the model returns result in the provided way.
 # 🎬 Sentiment Analysis Mini-Report

    Total test samples: 40
    Overall Accuracy: 60.00%

    Per-class results:
    Positive:
      True count:    13
      Predicted:     8
      Correct:       8
    Negative:
      True count:    11
      Predicted:     11
      Correct:       8
    Neutral:
      True count:    16
      Predicted:     8
      Correct:       8


# 📊 Confusion Matrix
                      Pred Positive  Pred Negative  Pred Neutral
    True Positive              8              0             0
    True Negative              0              8             0
    True Neutral               0              3             8

# 📈 Classification Report
              precision    recall  f1-score   support

    Positive       1.00      0.62      0.76        13
    Negative       0.73      0.73      0.73        11
     Neutral       1.00      0.50      0.67        16

    micro avg       0.89      0.60      0.72        40
    macro avg       0.91      0.61      0.72        40
    weighted avg    0.93      0.60      0.71        40


