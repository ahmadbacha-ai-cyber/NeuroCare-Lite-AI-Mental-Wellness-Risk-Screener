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
        "suicid", "suicide", "kill myself", "end it all", "want to die",
        "no reason to live", "no reason to live anymore",
        "better off dead", "hopeless", "worthless",
        "nothing matters", "end my life", "life is pointless",
        "life has no meaning", "wish i was dead", "wish i wasn't here",
        "i am a burden", "everyone will be better without me",
        "can't continue", "tired of living", "giving up on life",
        "i don't deserve to live", "wish i could disappear",
        "lost the will to live", "ending everything", "emotional death",
        "no purpose to life", "i want everything to stop",
        "life is unbearable", "can't live like this",
        "no way out", "trapped in pain", "nothing will ever get better",
        "over it all", "done with everything"
    ],
    "high": [
        "depressed", "can't go on", "giving up", "no hope",
        "empty inside", "numb", "don't care anymore", "pointless",
        "useless", "hate myself", "broken inside", "emotionally drained",
        "mentally exhausted", "dead inside", "can't feel anything",
        "no joy in life", "nothing excites me", "no motivation",
        "self-loathing", "self hate", "self-hate", "meaningless",
        "heartbroken", "deep sadness", "overly negative",
        "hopeless about future", "lost meaning",
        "so tired of everything", "emotionally numb",
        "lost myself", "feel broken", "permanently sad",
        "empty all the time", "mentally drained", "constant sadness"
    ],
    "moderate": [
        "sad", "down", "low mood", "crying", "unhappy", "miserable",
        "can't sleep", "insomnia", "exhausted", "no energy",
        "tired all the time", "fatigue", "low motivation",
        "can't focus", "brain fog", "mental fog", "feeling lonely",
        "isolated", "withdrawn", "lost interest", "don't enjoy",
        "unhappy with life", "low appetite", "overeating",
        "slow thinking", "feeling dull", "feeling empty",
        "feeling blue", "down most days", "can't enjoy hobbies",
        "no interest in friends", "socially withdrawn",
        "crying at night", "waking up tired",
        "dragging myself through the day"
    ],
    "mild": [
        "unmotivated", "lonely", "isolated", "withdrawn", "lost interest",
        "don't enjoy", "flat", "blah", "feeling off", "emotionally tired",
        "low enthusiasm", "low energy", "slightly down", "bored of life",
        "not feeling myself", "mild sadness", "low mood lately",
        "not in the mood for anything", "small sadness", "feeling low",
        "a bit down", "slightly low"
    ]
}

ANXIETY_KEYWORDS = {
    "severe": [
        "panic attack", "can't breathe", "heart racing", "going to die",
        "losing control", "terrified", "paralyzed with fear",
        "hyperventilating", "breathing fast", "shaking uncontrollably",
        "complete panic", "intense fear", "overwhelming dread",
        "mind spiraling", "freeze response", "extreme panic",
        "feeling unsafe", "terror", "sense of impending doom"
    ],
    "high": [
        "anxious", "panic", "anxiety", "terrified", "scared", "fear",
        "racing thoughts", "can't calm down", "overwhelming",
        "mind won't stop", "constant worry", "worried all the time",
        "catastrophizing", "fearful", "on high alert",
        "always afraid", "panic feeling", "can't relax",
        "high anxiety", "always on edge"
    ],
    "moderate": [
        "worried", "nervous", "on edge", "restless", "tense", "uneasy",
        "can't relax", "worried sick", "butterflies", "jittery",
        "tight chest", "shaky", "sweaty palms", "pressure in chest",
        "overthinking", "mind racing", "stomach tight", "tense muscles",
        "nervous energy", "easily startled"
    ],
    "mild": [
        "concerned", "apprehensive", "uncertain", "uncomfortable",
        "a bit worried", "slightly nervous", "mild anxiety",
        "small fear", "minor worry", "slight tension", "light worry",
        "a little anxious"
    ]
}

STRESS_KEYWORDS = {
    "severe": [
        "breaking point", "can't cope", "too much", "overwhelmed",
        "breaking down", "can't handle", "falling apart",
        "mentally collapsing", "total overload", "mental breakdown",
        "unmanageable stress", "emotionally overwhelmed",
        "shutting down", "too much pressure", "maxed out",
        "brain shutting down", "completely overwhelmed",
        "can't function anymore", "overwhelmed to the point of tears",
        "completely burned out", "emotionally collapsing"
    ],
    "high": [
        "stressed out", "burned out", "pressure", "overloaded",
        "can't keep up", "drowning", "crushed",
        "too many responsibilities", "high tension",
        "too demanding", "constant pressure", "non-stop stress",
        "chronic stress", "high stress", "always stressed",
        "constant workload", "piled up work"
    ],
    "moderate": [
        "stressed", "hectic", "busy", "demanding", "tiring",
        "struggling", "difficult", "challenging", "fatigued",
        "overworked", "emotionally tired", "mentally tired",
        "ongoing stress", "brain tired", "chaotic day",
        "stressful week", "under pressure", "draining day"
    ],
    "mild": [
        "a lot going on", "bit much", "stretched thin", "juggling",
        "slightly stressed", "some stress", "short-term stress",
        "handling a lot", "minor pressure", "a little stressed",
        "small amount of stress"
    ]
}

# Suicidality patterns – used to hard-flag high-risk text
SUICIDAL_PATTERNS = [
    "suicid", "suicide", "kill myself", "end it all", "want to die",
    "no reason to live", "no reason to live anymore",
    "life is pointless", "life has no meaning",
    "better off dead", "end my life",
    "wish i was dead", "wish i wasn't here",
    "i am a burden", "everyone will be better without me",
    "tired of living", "giving up on life",
    "i dont deserve to live", "i don't deserve to live",
    "wish i could disappear", "lost the will to live",
    "emotional death", "life is unbearable", "cant live like this",
    "can't live like this", "no way out", "trapped in pain",
    "nothing will ever get better", "over it all", "done with everything",
    "i want everything to stop"
]

# Positive / protective indicators
POSITIVE_INDICATORS = [
    "happy", "great", "good", "better", "improving", "hopeful",
    "excited", "joyful", "full of joy", "grateful", "thankful",
    "blessed", "peaceful", "calm", "relaxed", "confident", "strong",
    "proud", "optimistic", "motivated", "balanced", "content", "fine",
    "positive", "doing well", "feeling okay", "stable", "refreshed",
    "energized", "hopeful about future", "glad", "satisfied",
    "feeling strong", "feeling supported", "steady", "secure",
    "encouraged", "uplifted", "productive", "in control",
    "focused", "reassured", "comforted", "coping well",
    "managing okay", "feeling better", "feeling calm",
    "feeling peaceful"
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
#   ADVANCED KEYWORD-BASED ANALYZER (IMPROVED)
# =============================================================================
def _analyze_keywords(text: str):
    """
    Improved multi-dimensional keyword analysis.

    - Normalizes quotes, punctuation, spacing.
    - Uses stronger weights for severe indicators (esp. suicidality).
    - Handles basic negation (e.g., "not happy", "never sad").
    - Adds a suicidality flag for clearly high-risk content.
    - Combined risk = max(depression, anxiety, stress).
    """

    # --- Normalize text ---
    text_lower = text.lower()
    text_lower = (
        text_lower.replace("’", "'")
                  .replace("‘", "'")
                  .replace("“", '"')
                  .replace("”", '"')
                  .replace("…", " ")
    )

    # Remove punctuation for matching
    text_clean = re.sub(r"[^\w\s]", " ", text_lower)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    # Helper: count matches of a phrase with simple negation handling
    def count_matches(clean_text: str, phrase: str) -> int:
        """
        Returns how many times phrase appears in clean_text,
        ignoring occurrences that are clearly negated
        (e.g., 'not happy', 'never sad').
        """
        phrase_norm = phrase.lower()
        phrase_norm = re.sub(r"[^\w\s]", " ", phrase_norm)
        phrase_norm = re.sub(r"\s+", " ", phrase_norm).strip()

        if not phrase_norm:
            return 0

        count = 0
        start = 0
        while True:
            idx = clean_text.find(phrase_norm, start)
            if idx == -1:
                break

            # Look a bit before the phrase for a negation word
            window_start = max(0, idx - 20)
            prefix = clean_text[window_start:idx]

            if not re.search(r"\b(no|not|never|hardly)\b", prefix):
                count += 1

            start = idx + len(phrase_norm)
            if start >= len(clean_text):
                break

        return count

    # --- Scoring with improved weights ---
    def calculate_score(keywords_dict):
        score = 0.0

        for severity, words in keywords_dict.items():
            if severity == "severe":
                weight = 0.60
            elif severity == "high":
                weight = 0.35
            elif severity == "moderate":
                weight = 0.20
            else:
                weight = 0.10

            for phrase in words:
                hits = count_matches(text_clean, phrase)
                if hits > 0:
                    score += weight * hits

        return min(score, 1.0)

    # Individual scores
    depression_score = calculate_score(DEPRESSION_KEYWORDS)
    anxiety_score = calculate_score(ANXIETY_KEYWORDS)
    stress_score = calculate_score(STRESS_KEYWORDS)

    # --- Suicidality flag ---
    suicidal_flag = False
    for phrase in SUICIDAL_PATTERNS:
        if count_matches(text_clean, phrase) > 0:
            suicidal_flag = True
            break

    # If suicidality detected, enforce high depression risk
    if suicidal_flag:
        depression_score = max(depression_score, 0.9)

    # --- Positive indicators with negation handling ---
    positive_count = 0
    for word in POSITIVE_INDICATORS:
        if count_matches(text_clean, word) > 0:
            positive_count += 1

    # Positive modifier (capped)
    positive_modifier = 1.0 - min(positive_count * 0.07, 0.35)
    positive_modifier = max(0.5, positive_modifier)  # never reduce more than 50%

    depression_score *= positive_modifier
    anxiety_score *= positive_modifier
    stress_score *= positive_modifier

    # --- First-person emotional phrasing boost ---
    first_person_phrases = [
        "i feel", "i am", "i'm", "im", "i cant", "i can't",
        "i dont", "i don't", "i wish", "i want", "i have", "i hate"
    ]
    if any(p in text_clean for p in first_person_phrases):
        depression_score *= 1.15
        anxiety_score *= 1.15
        stress_score *= 1.15

    depression_score = min(depression_score, 1.0)
    anxiety_score = min(anxiety_score, 1.0)
    stress_score = min(stress_score, 1.0)

    # --- Combined risk ---
    combined_risk = max(depression_score, anxiety_score, stress_score)

    # --- Overall sentiment label ---
    if suicidal_flag or combined_risk >= 0.60:
        sentiment = "High Risk"
    elif combined_risk >= 0.30:
        sentiment = "Moderate Risk"
    elif positive_count >= 2 and combined_risk < 0.30:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    return {
        "overall_sentiment": sentiment,
        "depression_risk": round(depression_score, 2),
        "anxiety_risk": round(anxiety_score, 2),
        "stress_risk": round(stress_score, 2),
        "combined_risk": round(combined_risk, 2),
        "suicidal_flag": suicidal_flag
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
    3. Combines both for best results, with keyword override
       for clearly high-risk suicidal content.
    """
    if not text or len(text.strip().split()) < 2:
        return {
            "overall_sentiment": "Neutral",
            "depression_risk": 0.0,
            "anxiety_risk": 0.0,
            "stress_risk": 0.0,
            "combined_risk": 0.0
        }

    # 1) Keyword analysis (always)
    keyword_results = _analyze_keywords(text)
    suicidal_flag = keyword_results.get("suicidal_flag", False)

    # Default: if transformer fails, use keyword-only
    final_results = keyword_results

    # 2) Transformer-based analysis
    try:
        tokenizer, model = load_model(local_files_only=allow_local_files_only)
        transformer_results = _transformer_analysis(text, tokenizer, model)

        # Weighting: trust keywords more, especially in high-risk cases
        alpha = 0.7  # keyword weight
        if suicidal_flag or keyword_results["combined_risk"] >= 0.75:
            alpha = 0.9  # heavily trust keywords

        dep = keyword_results["depression_risk"] * alpha + transformer_results["depression_risk"] * (1 - alpha)
        anx = keyword_results["anxiety_risk"] * alpha + transformer_results["anxiety_risk"] * (1 - alpha)
        st =  keyword_results["stress_risk"] * alpha + transformer_results["stress_risk"] * (1 - alpha)

        mix_combined = keyword_results["combined_risk"] * alpha + transformer_results["combined_risk"] * (1 - alpha)
        combined = max(mix_combined, keyword_results["combined_risk"])

        if suicidal_flag:
            dep = max(dep, 0.9)
            combined = max(combined, 0.9)

        final_results = {
            "overall_sentiment": keyword_results["overall_sentiment"],
            "depression_risk": round(dep, 2),
            "anxiety_risk": round(anx, 2),
            "stress_risk": round(st, 2),
            "combined_risk": round(combined, 2)
        }

    except Exception as e:
        logger.info(f"[NLP] Transformer unavailable, using keyword analysis only: {e}")
        final_results = keyword_results

    return final_results

# =============================================================================
#   LOCAL TEST MODE
# =============================================================================
if __name__ == "__main__":
    test_cases = [
        "I feel great today and I'm really happy with how things are going!",
        "I'm so stressed out and anxious about everything. Can't sleep at night.",
        "I wish I could disappear. I don't see any reason to live anymore.",
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
        print(
            f"Depression: {result['depression_risk']:.0%} | "
            f"Anxiety: {result['anxiety_risk']:.0%} | "
            f"Stress: {result['stress_risk']:.0%}"
        )
        print(f"Combined Risk: {result['combined_risk']:.0%}")
        print("-" * 70)
