#!/usr/bin/env python3
# ==========================================
# Sentiment Analyzer
# Last experimented: 2025-11-18
# Platform: M1 Mac
# ==========================================
#
# Uses HuggingFace 'transformers' library.
#
# Analyzes sentiment for multiple text samples.
#       Each returns:
#           - Sentiment label (POSITIVE / NEGATIVE)
#           - Confidence score
#
# Setup:
#       python3 -m venv venv
#       source venv/bin/activate
#       pip install torch transformers
#
# Run:
#       python sentiment.py
#       python sentiment.py "I feel great today." "Do I feel great today?"
#
# Notes:
#       The first run will download the model.
#       Subsequent runs are fast thanks to local caching.
#
# License:
#       MIT License
#
# Make it positive ⤴️
# ==========================================

from transformers import pipeline
import sys

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)


def run_analysis(texts):
    print("=== SENTIMENT RESULTS ===\n")
    for idx, t in enumerate(texts, start=1):

        try:
            result = analyzer(t)[0]
            label = result["label"]
            score = round(result["score"], 4)
        except Exception as e:
            print(f"{idx}. {t}")
            print(f"   !! Error: {e}\n")
            continue

        print(f"{idx}. {t}")
        print(f"   -> {label} ({score})\n")
    print("==========================")


def main():
    if len(sys.argv) > 1:
        texts = sys.argv[1:]
    else:
        texts = [
            # Inspired by Terminator 2
            "He walked out of the fire like nothing could stop him.",
            # Inspired by Star Wars
            "I've got a strange feeling the universe is about to test me again.",
            # Inspired by The Dark Knight
            "He smiled like he enjoyed watching the world wobble on the edge.",
            # Inspired by Toy Story
            "The old toy looked up with hope, ready for one more adventure.",
            # Inspired by Titanic
            "She stood at the railing, convinced the night could hold her dreams.",
            # Inspired by Lord of the Rings
            "The journey felt impossible, but he kept walking because someone had to.",
            # Inspired by The Matrix
            "Reality flickered for a moment, as if questioning its own shape.",
            # Inspired by Finding Nemo
            "He swam forward even though the ocean still scared him.",
            # Inspired by Spider-Man
            "The city looked different from above, heavier now that he had responsibility.",
            # Inspired by Jurassic Park
            "The ground trembled softly, like something huge had just decided to wake up.",
        ]

    run_analysis(texts)


if __name__ == "__main__":
    main()
