# nlp_screener.py
import torch
import sys
import platform
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from scipy.special import softmax
import logging
import re

logger = logging.getLogger(__name__)

# =============================================================================
#   MODEL CONFIG
# =============================================================================
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# =============================================================================
#   MENTAL HEALTH KEYWORD DICTIONARIES (Evidence-Based)
# =============================================================================
DEPRESSION_KEYWORDS = {
    "severe": [
        "suicid", "kill myself", "end it all", "want to die", "no reason to live",
        "better off dead", "hopeless", "worthless", "nothing matters"
    ],
    "high": [
        "depressed", "can't go on", "giving up", "no hope", "empty inside",
        "numb", "don't care anymore", "pointless", "useless", "hate myself"
    ],
    "moderate": [
        "sad", "down", "low mood", "crying", "unhappy", "miserable",
        "can't sleep", "insomnia", "exhausted", "no energy", "tired all the time"
    ],
    "mild": [
        "unmotivated", "lonely", "isolated", "withdrawn", "lost interest",
        "don't enjoy", "flat", "blah"
    ]
}

ANXIETY_KEYWORDS = {
    "severe": [
        "panic attack", "can't breathe", "heart racing", "going to die",
        "losing control", "terrified", "paralyzed with fear"
    ],
    "high": [
        "anxious", "panic", "anxiety", "terrified", "scared", "fear",
        "racing thoughts", "can't calm down", "overwhelming"
    ],
    "moderate": [
        "worried", "nervous", "on edge", "restless", "tense", "uneasy",
        "can't relax", "worried sick", "butterflies", "jittery"
    ],
    "mild": [
        "concerned", "apprehensive", "uncertain", "uncomfortable",
        "a bit worried", "slightly nervous"
    ]
}

STRESS_KEYWORDS = {
    "severe": [
        "breaking point", "can't cope", "too much", "overwhelmed",
        "breaking down", "can't handle", "falling apart"
    ],
    "high": [
        "stressed out", "burned out", "pressure", "overloaded",
        "can't keep up", "drowning", "crushed"
    ],
    "moderate": [
        "stressed", "hectic", "busy", "demanding", "tiring",
        "struggling", "difficult", "challenging"
    ],
    "mild": [
        "a lot going on", "bit much", "stretched thin", "juggling"
    ]
}

POSITIVE_INDICATORS = [
    "happy", "great", "good", "better", "improving", "hopeful",
    "excited", "joy", "grateful", "thankful", "blessed", "peaceful",
    "calm", "relaxed", "confident", "strong", "proud"
]

# =============================================================================
#   SAFE LAZY MODEL LOADER
# =============================================================================
@st.cache_resource
def load_model(local_files_only: bool = False):
    """Loads tokenizer & model lazily and safely."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            local_files_only=local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            local_files_only=local_files_only
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.warning(f"[NLP] HF model load failed: {e}")
        raise

# =============================================================================
#   ADVANCED KEYWORD-BASED ANALYZER
# =============================================================================
def _analyze_keywords(text: str):
    """
    Multi-dimensional mental health keyword analysis.
    Returns individual risk scores for depression, anxiety, and stress.
    """
    text_lower = text.lower()
    
    # Remove punctuation for better matching
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
    
    # Score calculation function
    def calculate_score(keywords_dict):
        score = 0.0
        for severity, words in keywords_dict.items():
            for word in words:
                if word in text_clean:
                    if severity == "severe":
                        score += 0.25
                    elif severity == "high":
                        score += 0.18
                    elif severity == "moderate":
                        score += 0.12
                    elif severity == "mild":
                        score += 0.08
        return min(score, 1.0)
    
    # Calculate individual scores
    depression_score = calculate_score(DEPRESSION_KEYWORDS)
    anxiety_score = calculate_score(ANXIETY_KEYWORDS)
    stress_score = calculate_score(STRESS_KEYWORDS)
    
    # Check for positive indicators (reduces risk)
    positive_count = sum(1 for word in POSITIVE_INDICATORS if word in text_clean)
    positive_modifier = max(0.0, 1.0 - (positive_count * 0.1))
    
    # Apply positive modifier
    depression_score *= positive_modifier
    anxiety_score *= positive_modifier
    stress_score *= positive_modifier
    
    # Boost scores for first-person distress
    first_person_phrases = ["i feel", "i am", "i can't", "i'm", "i have"]
    if any(phrase in text_lower for phrase in first_person_phrases):
        depression_score *= 1.2
        anxiety_score *= 1.2
        stress_score *= 1.2
    
    # Ensure scores don't exceed 1.0
    depression_score = min(depression_score, 1.0)
    anxiety_score = min(anxiety_score, 1.0)
    stress_score = min(stress_score, 1.0)
    
    # Combined risk (weighted average)
    combined_risk = (depression_score * 0.4 + anxiety_score * 0.35 + stress_score * 0.25)
    
    # Determine overall sentiment
    if combined_risk >= 0.6:
        sentiment = "High Risk"
    elif combined_risk >= 0.3:
        sentiment = "Moderate Risk"
    elif positive_count > 2:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"
    
    return {
        "overall_sentiment": sentiment,
        "depression_risk": round(depression_score, 2),
        "anxiety_risk": round(anxiety_score, 2),
        "stress_risk": round(stress_score, 2),
        "combined_risk": round(combined_risk, 2)
    }

# =============================================================================
#   TRANSFORMER-BASED SENTIMENT ANALYSIS (ENHANCED)
# =============================================================================
def _transformer_analysis(text: str, tokenizer, model):
    """
    Uses transformer model for sentiment analysis,
    then maps to mental health risk scores.
    """
    model_max = getattr(model.config, "max_position_embeddings", 512)
    tokenizer_max = getattr(tokenizer, "model_max_length", 512)
    effective_max = min(model_max, tokenizer_max, 1024)

    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=effective_max,
        add_special_tokens=True
    )

    # Ensure sequence length is within model limits
    seq_len = encoded_input["input_ids"].shape[-1]
    if seq_len > model_max:
        encoded_input = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=model_max,
            add_special_tokens=True
        )

    # Inference
    with torch.no_grad():
        output = model(**encoded_input)

    logits = output.logits[0].cpu().numpy()
    probs = softmax(logits)

    labels = ["Negative", "Neutral", "Positive"]
    idx = int(logits.argmax())
    sentiment = labels[idx]
    
    # Map sentiment to risk scores
    if sentiment == "Negative":
        base_risk = float(probs[0])
        return {
            "overall_sentiment": sentiment,
            "depression_risk": round(base_risk * 0.9, 2),
            "anxiety_risk": round(base_risk * 0.85, 2),
            "stress_risk": round(base_risk * 0.8, 2),
            "combined_risk": round(base_risk, 2)
        }
    elif sentiment == "Neutral":
        return {
            "overall_sentiment": sentiment,
            "depression_risk": 0.15,
            "anxiety_risk": 0.15,
            "stress_risk": 0.15,
            "combined_risk": 0.15
        }
    else:  # Positive
        return {
            "overall_sentiment": sentiment,
            "depression_risk": 0.0,
            "anxiety_risk": 0.0,
            "stress_risk": 0.0,
            "combined_risk": 0.0
        }

# =============================================================================
#   MAIN ANALYSIS FUNCTION (HYBRID APPROACH)
# =============================================================================
def analyze_text(text: str, allow_local_files_only: bool = False):
    """
    Main text analysis function using hybrid approach:
    1. Keyword-based analysis (reliable, interpretable)
    2. Transformer-based sentiment (contextual understanding)
    3. Combines both for best results
    """
    if not text or len(text.strip().split()) < 2:
        return {
            "overall_sentiment": "Neutral",
            "depression_risk": 0.0,
            "anxiety_risk": 0.0,
            "stress_risk": 0.0,
            "combined_risk": 0.0
        }

    # Get keyword-based analysis (always runs)
    keyword_results = _analyze_keywords(text)
    
    # Try to get transformer-based analysis
    try:
        tokenizer, model = load_model(local_files_only=allow_local_files_only)
        transformer_results = _transformer_analysis(text, tokenizer, model)
        
        # Combine results (keyword analysis weighted higher for reliability)
        combined_results = {
            "overall_sentiment": keyword_results["overall_sentiment"],
            "depression_risk": round(
                keyword_results["depression_risk"] * 0.7 + transformer_results["depression_risk"] * 0.3,
                2
            ),
            "anxiety_risk": round(
                keyword_results["anxiety_risk"] * 0.7 + transformer_results["anxiety_risk"] * 0.3,
                2
            ),
            "stress_risk": round(
                keyword_results["stress_risk"] * 0.7 + transformer_results["stress_risk"] * 0.3,
                2
            ),
            "combined_risk": round(
                keyword_results["combined_risk"] * 0.7 + transformer_results["combined_risk"] * 0.3,
                2
            )
        }
        return combined_results
        
    except Exception as e:
        logger.info(f"[NLP] Transformer unavailable, using keyword analysis only: {e}")
        return keyword_results

# =============================================================================
#   LOCAL TEST MODE
# =============================================================================
if __name__ == "__main__":
    test_cases = [
        "I feel great today and I'm really happy with how things are going!",
        "I'm so stressed out and anxious about everything. Can't sleep at night.",
        "I feel hopeless and don't see any point in going on. Everything is worthless.",
        "Just having a normal day, nothing special.",
        "I'm a bit worried about the presentation tomorrow but I'll be fine.",
        "I can't breathe, having panic attacks constantly, terrified all the time.",
        "Feeling a bit down and tired lately, lost interest in things I used to enjoy."
    ]
    
    print("=" * 70)
    print("MENTAL HEALTH TEXT ANALYSIS TEST")
    print("=" * 70)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {text}")
        result = analyze_text(text)
        print(f"Sentiment: {result['overall_sentiment']}")
        print(f"Depression: {result['depression_risk']:.0%} | Anxiety: {result['anxiety_risk']:.0%} | Stress: {result['stress_risk']:.0%}")
        print(f"Combined Risk: {result['combined_risk']:.0%}")
        print("-" * 70)