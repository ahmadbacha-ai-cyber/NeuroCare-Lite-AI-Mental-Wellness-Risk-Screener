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
        # --- Core suicidality ---
        "suicid", "suicide", "kill myself", "end it all", "want to die",
        "better off dead", "wish i was dead", "wish i wasn't here",
        "no reason to live", "life is pointless", "life has no meaning",
        "i hope i don’t wake up", "i want everything to stop",
        "i'm done with life", "everyone would be better without me",
        "lost the will to live", "final goodbye", "ending everything",
        "can’t continue anymore", "can’t live like this",
        "my life is over", "i want to disappear", "death would be easier",
        "escape life", "nothing will ever improve",
        "my existence is pointless", "total despair", "i don’t deserve to live",

        # --- Catastrophic hopelessness ---
        "utterly hopeless", "completely hopeless", "permanent despair",
        "no future", "hopeless future", "future is dark",
        "never getting better", "beyond saving", "helpless forever",

        # --- Severe depressive cognition ---
        "total emptiness", "emotionally dead", "mentally collapsing",
        "can’t feel anything", "absolute numbness", "mentally destroyed",

        # --- Severe anhedonia ---
        "i feel nothing at all", "no pleasure in anything",
        "zero motivation", "nothing matters anymore",

        # --- Functional collapse ---
        "can’t function at all", "unable to get out of bed",
        "completely shut down", "mentally broken", "severe breakdown"
    ],

    "high": [
        # --- Intense negative emotions ---
        "deep sadness", "emotionally drained", "dead inside",
        "mentally exhausted", "emotionally numb", "self-hate",
        "self loathing", "hate myself", "broken inside", "constant sadness",
        "continuous despair", "emotionally empty",

        # --- Major anhedonia ---
        "nothing excites me", "no motivation at all",
        "no joy", "lost all interest", "can't feel joy",

        # --- Cognitive impairment ---
        "constant brain fog", "can't think straight", "slow thinking",
        "mentally stuck", "mental heaviness", "dark thoughts",

        # --- Behavioral signs ---
        "barely functioning", "stopped caring", "can’t manage life",
        "barely surviving", "isolating completely"
    ],

    "moderate": [
        # --- Classic depression symptoms ---
        "sad", "down", "low mood", "crying", "miserable", "unhappy",
        "insomnia", "hypersomnia", "fatigue", "low energy", "tired all the time",
        "lost interest", "don’t enjoy things", "feeling lonely",
        "isolated", "withdrawn", "no appetite", "overeating",
        "brain fog", "feeling blue", "dragging myself", "slowed down",
        "unmotivated", "emotional fatigue", "low confidence",
        "feeling hopeless", "feeling worthless", "empty inside"
    ],

    "mild": [
        # --- Subclinical symptoms ---
        "a bit down", "slightly sad", "mild sadness",
        "feeling off", "blah", "flat", "low enthusiasm",
        "not feeling myself", "emotionally tired",
        "feeling low", "slightly empty", "bored with life",
        "low mood lately"
    ]
}

ANXIETY_KEYWORDS = {
    "severe": [
        "panic attack", "can't breathe", "hyperventilating",
        "heart racing fast", "losing control", "paralyzed with fear",
        "intense terror", "sense of impending doom",
        "shaking uncontrollably", "complete panic",
        "can't calm down at all", "severe dread", "freeze response",
        "catastrophic fear", "irrational terror", "mental paralysis"
    ],

    "high": [
        "panic", "very anxious", "terrified", "scared", "fearful",
        "racing thoughts", "mind won't stop", "constant worry",
        "catastrophizing", "always afraid", "high anxiety",
        "on high alert", "overwhelming worry", "fear spiraling",
        "obsessing", "ruminating", "intrusive thoughts"
    ],

    "moderate": [
        "worried", "nervous", "restless", "uneasy", "tense",
        "jittery", "shaky", "sweaty palms", "tight chest",
        "mind racing", "pressure in chest", "overthinking",
        "tense muscles", "easily startled", "fear building up",
        "feeling overwhelmed", "mild dread"
    ],

    "mild": [
        "a bit worried", "slightly anxious", "mild anxiety",
        "light worry", "uncertain", "apprehensive",
        "slight nervousness", "small fear", "uneasy feeling"
    ]
}

STRESS_KEYWORDS = {
    "severe": [
        "breaking point", "can't cope", "overwhelmed completely",
        "falling apart", "total overload", "complete burnout",
        "emotionally collapsing", "maxed out", "can't function anymore",
        "total mental breakdown", "shutdown mode",
        "too much to handle", "losing control from stress",
        "unmanageable stress", "overwhelmed to tears",
        "completely drowned in stress"
    ],

    "high": [
        "stressed out", "burned out", "pressure", "overloaded",
        "too much work", "constant pressure", "chronic stress",
        "high stress", "always stressed", "non-stop stress",
        "constant workload", "piled up work", "crushed by stress",
        "too demanding", "constant tension", "emotional overload"
    ],

    "moderate": [
        "stressed", "hectic", "tiring", "challenging", "difficult",
        "fatigued", "overworked", "draining day", "stressful week",
        "mentally tired", "emotionally tired", "brain tired",
        "demanding day", "busy and stressed"
    ],

    "mild": [
        "a lot going on", "slightly stressed", "some stress",
        "minor pressure", "a little stressed", "short-term stress",
        "handling a lot", "bit overwhelmed"
    ]
}


# Suicidality patterns – used to hard-flag high-risk text
SUICIDAL_PATTERNS = [
    # Direct
    "suicide", "kill myself", "end my life", "i want to die",
    "better off dead", "wish i was dead", "wish i wasn't here",

    # Indirect
    "no reason to live", "i can't do this anymore", "done with life",
    "tired of living", "giving up on life", "life isn't worth it",
    "everyone would be better without me",

    # Coded suicide language
    "ready to go", "won't be here tomorrow", "final goodbye",
    "i hope i don’t wake up", "i'm finished", "i'm done here",

    # Existential despair
    "my existence is pointless", "nothing will ever improve",
    "utter hopelessness", "permanent despair"
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

    # Normalize common suicide phrase typos/splits
    # e.g., "kill my self" -> "kill myself"
    text_clean = text_clean.replace("my self", "myself")

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
        "suicidal_flag": suicidal_flag,
        "positive_count": int(positive_count),
        "token_count": len(text_clean.split())
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
            "combined_risk": 0.0,
            "suicidal_flag": False,
            "analysis_confidence": 0.3,
            "primary_concern": "No Strong Signal"
        }

    # 1) Keyword analysis (always)
    keyword_results = _analyze_keywords(text)
    suicidal_flag = keyword_results.get("suicidal_flag", False)
    token_count = keyword_results.get("token_count", len(text.split()))
    keyword_combined = keyword_results["combined_risk"]

    # Default final_results = keyword-only
    final_results = keyword_results

    # 2) Transformer-based analysis
    transformer_results = None
    try:
        tokenizer, model = load_model(local_files_only=allow_local_files_only)
        transformer_results = _transformer_analysis(text, tokenizer, model)

        # Weighting: trust keywords more, especially in high-risk cases
        alpha = 0.7  # keyword weight
        if suicidal_flag or keyword_combined >= 0.75:
            alpha = 0.9  # heavily trust keywords

        dep = keyword_results["depression_risk"] * alpha + transformer_results["depression_risk"] * (1 - alpha)
        anx = keyword_results["anxiety_risk"] * alpha + transformer_results["anxiety_risk"] * (1 - alpha)
        st_risk = keyword_results["stress_risk"] * alpha + transformer_results["stress_risk"] * (1 - alpha)

        mix_combined = keyword_combined * alpha + transformer_results["combined_risk"] * (1 - alpha)
        combined = max(mix_combined, keyword_combined)

        if suicidal_flag:
            dep = max(dep, 0.9)
            combined = max(combined, 0.9)

        final_results = {
            "overall_sentiment": keyword_results["overall_sentiment"],
            "depression_risk": round(dep, 2),
            "anxiety_risk": round(anx, 2),
            "stress_risk": round(st_risk, 2),
            "combined_risk": round(combined, 2),
            "suicidal_flag": suicidal_flag
        }

    except Exception as e:
        logger.info(f"[NLP] Transformer unavailable, using keyword analysis only: {e}")
        final_results = {
            "overall_sentiment": keyword_results["overall_sentiment"],
            "depression_risk": keyword_results["depression_risk"],
            "anxiety_risk": keyword_results["anxiety_risk"],
            "stress_risk": keyword_results["stress_risk"],
            "combined_risk": keyword_results["combined_risk"],
            "suicidal_flag": suicidal_flag
        }

    # 3) Primary concern label
    dep = final_results["depression_risk"]
    anx = final_results["anxiety_risk"]
    st_risk = final_results["stress_risk"]

    scores = {"Depression": dep, "Anxiety": anx, "Stress": st_risk}
    main_label = max(scores, key=scores.get)
    max_score = scores[main_label]

    # Decide if it's clearly focused or mixed/none
    sorted_scores = sorted(scores.values(), reverse=True)
    if max_score < 0.20:
        primary_concern = "No Strong Signal"
    elif len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) < 0.10:
        primary_concern = "Mixed (Depression / Anxiety / Stress)"
    else:
        primary_concern = main_label

    # 4) Analysis confidence (0–1)
    #    Based on text length + combined risk + suicidality override
    length_factor = min(token_count / 35.0, 1.0)  # saturates around ~35 tokens
    risk_factor = float(final_results["combined_risk"])
    conf = 0.25 + 0.45 * length_factor + 0.30 * risk_factor
    if suicidal_flag:
        conf = max(conf, 0.85)
    conf = max(0.0, min(1.0, conf))

    final_results["primary_concern"] = primary_concern
    final_results["analysis_confidence"] = round(conf, 2)

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
        "Feeling a bit down and tired lately, lost interest in things I used to enjoy.",
        "today i not feel good i think i will kill my self"
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
        print(f"Primary Concern: {result.get('primary_concern')}")
        print(f"Confidence: {result.get('analysis_confidence')}")
        print(f"Suicidal Flag: {result.get('suicidal_flag')}")
        print("-" * 70)
