import pandas as pd
from collections import Counter
from sentiment_llm import SentimentAnalyzer   
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_on_csv(csv_path, analyzer, review_col="review", label_col="sentiment", sample_size=50):
    """
    Evaluate sentiment analysis on a sample of reviews from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        analyzer (SentimentAnalyzer): Your analyzer instance.
        review_col (str): Column name with review text.
        label_col (str): Column name with true sentiment labels.
        sample_size (int): Number of rows to sample for evaluation.
    """
    # Load CSV dynamically
    df = pd.read_csv(csv_path)
    
    # Take a random sample (to keep evaluation small)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    results = []
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        # Normalize labels
        true_label = str(row[label_col]).strip().capitalize()
        review_text = str(row[review_col])

        pred = analyzer.analyze_sentiment(review_text)
        pred_label = pred["label"].capitalize()

        results.append((true_label, pred_label))
        y_true.append(true_label)
        y_pred.append(pred_label)

    # Accuracy
    correct = sum(1 for t, p in results if t == p)
    accuracy = correct / len(results) if results else 0

    # Per-class counts
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    correct_counts = Counter([t for t, p in results if t == p])

    print("\n🎬 Sentiment Analysis Mini-Report")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Total test samples: {len(results)}")
    print(f"Overall Accuracy: {accuracy:.2%}\n")

    print("Per-class results:")
    for label in ["Positive", "Negative", "Neutral"]:
        print(f"  {label}:")
        print(f"    True count:    {true_counts[label]}")
        print(f"    Predicted:     {pred_counts[label]}")
        print(f"    Correct:       {correct_counts[label]}")
    print("=" * 60)

    # Confusion Matrix
    print("\n📊 Confusion Matrix")
    labels = ["Positive", "Negative", "Neutral"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels],
                            columns=[f"Pred {l}" for l in labels])
    print(cm_df)

    # Classification report (precision, recall, F1)
    print("\n📈 Classification Report")
    print(classification_report(y_true, y_pred, labels=labels))

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    evaluate_on_csv(
        r"C:\Users\Sahan Ganguly\OneDrive\Desktop\Anthropod V2\movie_reviews.csv",
        analyzer,
        review_col="review",      
        label_col="sentiment",    
        sample_size=40            
    )
