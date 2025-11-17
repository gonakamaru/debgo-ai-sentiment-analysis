#!/usr/bin/env python3
# ==========================================
# Sentiment Analyzer
# Last experimented: 2025-11-17
# Platform: M1 Mac
# ==========================================
#
# Uses HuggingFace 'transformers' library.
#
# Analyzes sentiment for multiple text samples.
#   Each returns:
#       - Sentiment label (POSITIVE / NEGATIVE)
#       - Confidence score
#
# Run:
#   python sentiment.py
#
# Setup:
#   1. Create and activate a virtual environment
#        python3 -m venv venv
#        source venv/bin/activate
#
#   2. Install dependencies (choose one)
#        a) Installing manually:
#              pip install torch transformers
#
#        b) Using requirements.txt:
#              pip install -r requirements.txt
#
#   3. Run the script:
#        python sentiment.py
#
# Notes:
#   • The first run will download the model.
#   • Subsequent runs are fast thanks to local caching.
#
# License:
#   MIT License
#
# Make it positive ⤴️
# ==========================================

from transformers import pipeline


def main():

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    analyzer = pipeline("sentiment-analysis", model=model_name)

    texts = [
        "I really love how this project turned out.",
        "The weather has been terrible all week.",
        "The weather has been terrible all week except for today.",
        "I'm not sure how I feel about this…",
        "This is absolutely amazing!",
        "The service today was disappointing.",
        "This is an apple.",
        "This is just an apple.",
        "This apple is red.",
        "This apple is brown.",
        "This apple is green.",
    ]

    print("=== SENTIMENT RESULTS ===\n")

    for idx, t in enumerate(texts, start=1):
        result = analyzer(t)[0]
        label = result["label"]
        score = round(result["score"], 4)
        print(f"{idx}. {t}")
        print(f"   -> {label} ({score})\n")

    print("==========================")


if __name__ == "__main__":
    main()
