# voice_detector.py
import numpy as np
import librosa
import soundfile as sf
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
#   CROSS-PLATFORM AUDIO LOADER
# =============================================================================
def _safe_load_audio(audio_path, target_sr=22050):
    """
    Cross-platform safe audio loader.
    Handles macOS CoreAudio issues, Windows libsndfile issues, MP3 decoding.
    """
    try:
        # Try soundfile first (faster, cleaner)
        y, sr = sf.read(audio_path)
        if y.ndim > 1:  # stereo → mono
            y = y.mean(axis=1)
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        return y, target_sr
    except Exception:
        # Fall back to librosa loader (uses audioread)
        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return y, sr
        except Exception:
            return None, None

# =============================================================================
#   COMPREHENSIVE VOICE FEATURE EXTRACTION
# =============================================================================
def _extract_voice_features(y, sr):
    """
    Extract comprehensive voice features:
    - RMS Energy (volume/loudness)
    - Pitch (F0) using pYIN
    - Speech rate / tempo
    - Pause patterns
    - Pitch variation (vocal stability)
    - Zero-crossing rate (voice quality)
    """
    if y is None or len(y) < sr * 0.3:
        return None  # Audio too short

    features = {}

    # 1) RMS Energy (Volume Level)
    rms = librosa.feature.rms(y=y)[0]
    features['rms'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))

    # 2) Pitch Analysis (F0) using pYIN
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            sr=sr,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        # Clean F0 (remove NaN)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features['pitch'] = float(np.mean(f0_clean))
            features['pitch_std'] = float(np.std(f0_clean))
            features['pitch_min'] = float(np.min(f0_clean))
            features['pitch_max'] = float(np.max(f0_clean))
        else:
            features['pitch'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
    except Exception:
        features['pitch'] = 0.0
        features['pitch_std'] = 0.0
        features['pitch_min'] = 0.0
        features['pitch_max'] = 0.0

    # 3) Speech Rate Proxy
    threshold = np.mean(rms) * 0.1
    non_silent_frames = np.where(rms > threshold)[0]
    features['speech_rate'] = float(len(non_silent_frames) / len(rms))
    features['pause_ratio'] = float(1.0 - features['speech_rate'])

    # 4) Zero Crossing Rate (voice quality indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr'] = float(np.mean(zcr))

    # 5) Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid'] = float(np.mean(spectral_centroids))

    return features

# =============================================================================
#   VOLUME LEVEL CLASSIFICATION
# =============================================================================
def _classify_volume(rms, rms_max):
    """
    Classifies volume level: Quiet, Normal, Loud, Very Loud
    """
    if rms_max > 0.15:
        return "Very Loud", 0.8  # High stress indicator
    elif rms_max > 0.08:
        return "Loud", 0.5
    elif rms > 0.03:
        return "Normal", 0.2
    else:
        return "Quiet", 0.6  # Can indicate depression/low energy

# =============================================================================
#   SPEECH PACE CLASSIFICATION
# =============================================================================
def _classify_speech_pace(speech_rate, pause_ratio):
    """
    Classifies speech pace: Very Slow, Slow, Normal, Fast, Very Fast
    """
    if speech_rate < 0.40:
        return "Very Slow", 0.9  # Strong depression/fatigue indicator
    elif speech_rate < 0.60:
        return "Slow", 0.6  # Mild depression indicator
    elif speech_rate < 0.80:
        return "Normal", 0.2
    elif speech_rate < 0.92:
        return "Fast", 0.5  # Moderate anxiety indicator
    else:
        return "Very Fast", 0.8  # High anxiety indicator

# =============================================================================
#   PITCH-BASED STRESS DETECTION
# =============================================================================
def _analyze_pitch_stress(pitch, pitch_std, pitch_max):
    """
    Analyzes pitch characteristics for stress indicators:
    - High pitch = vocal tension (anxiety/stress)
    - High pitch variability = emotional instability
    - Monotone (low variation) = depression
    """
    stress_score = 0.0
    
    # High average pitch (vocal tension)
    if pitch > 220:
        stress_score += 0.35
    elif pitch > 180:
        stress_score += 0.25
    elif pitch > 150:
        stress_score += 0.15
    
    # Pitch variability
    if pitch_std > 50:
        stress_score += 0.25  # High variability = emotional instability
    elif pitch_std < 15 and pitch > 0:
        stress_score += 0.20  # Monotone = possible depression
    
    # Pitch range (max pitch)
    if pitch_max > 300:
        stress_score += 0.20  # Very high peaks = stress
    
    return min(stress_score, 1.0)

# =============================================================================
#   MAIN VOICE ANALYSIS FUNCTION
# =============================================================================
def analyze_voice(audio_path):
    """
    Comprehensive voice stress analyzer.
    Detects:
    - Volume level (quiet, normal, loud, very loud)
    - Speech pace (very slow, slow, normal, fast, very fast)
    - Pitch-based stress indicators
    - Overall voice-based mental health risk
    
    Returns: {
        "stress_level": str,
        "volume_level": str,
        "speech_pace": str,
        "voice_risk": float,
        "features": dict
    }
    """
    if not os.path.exists(audio_path):
        return {
            "stress_level": "No Audio File",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # Load audio safely
    y, sr = _safe_load_audio(audio_path)
    if y is None:
        return {
            "stress_level": "Audio Decode Error",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # Extract features
    features = _extract_voice_features(y, sr)
    if features is None:
        return {
            "stress_level": "Insufficient Audio",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # =========================================================================
    #   MULTI-DIMENSIONAL VOICE ANALYSIS
    # =========================================================================

    # 1) Volume Analysis
    volume_level, volume_risk = _classify_volume(
        features['rms'], 
        features['rms_max']
    )

    # 2) Speech Pace Analysis
    speech_pace, pace_risk = _classify_speech_pace(
        features['speech_rate'],
        features['pause_ratio']
    )

    # 3) Pitch-Based Stress Analysis
    pitch_risk = _analyze_pitch_stress(
        features['pitch'],
        features['pitch_std'],
        features['pitch_max']
    )

    # 4) Energy Variability (emotional instability indicator)
    rms_variation_risk = 0.0
    if features['rms_std'] > 0.03:
        rms_variation_risk = 0.3  # High energy variability

    # =========================================================================
    #   COMBINED VOICE RISK SCORE
    # =========================================================================
    # Weighted combination of all risk factors
    combined_risk = (
        volume_risk * 0.25 +
        pace_risk * 0.35 +
        pitch_risk * 0.30 +
        rms_variation_risk * 0.10
    )
    
    combined_risk = min(combined_risk, 1.0)

    # Determine overall stress level
    if combined_risk >= 0.70:
        stress_level = "Very High Stress"
    elif combined_risk >= 0.50:
        stress_level = "High Stress"
    elif combined_risk >= 0.30:
        stress_level = "Moderate Stress"
    else:
        stress_level = "Low Stress"

    # =========================================================================
    #   RETURN COMPREHENSIVE RESULTS
    # =========================================================================
    return {
        "stress_level": stress_level,
        "volume_level": volume_level,
        "speech_pace": speech_pace,
        "voice_risk": round(float(combined_risk), 2),
        "features": {
            "rms": features['rms'],
            "rms_std": features['rms_std'],
            "rms_max": features['rms_max'],
            "pitch": features['pitch'],
            "pitch_std": features['pitch_std'],
            "pitch_min": features['pitch_min'],
            "pitch_max": features['pitch_max'],
            "speech_rate": features['speech_rate'],
            "pause_ratio": features['pause_ratio'],
            "zcr": features['zcr'],
            "spectral_centroid": features['spectral_centroid']
        }
    }

# =============================================================================
#   LOCAL TEST MODE
# =============================================================================
if __name__ == "__main__":
    test_file = "test.wav"
    
    print("=" * 70)
    print("VOICE STRESS ANALYSIS TEST")
    print("=" * 70)
    
    if os.path.exists(test_file):
        result = analyze_voice(test_file)
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"  Stress Level: {result['stress_level']}")
        print(f"  Volume Level: {result['volume_level']}")
        print(f"  Speech Pace: {result['speech_pace']}")
        print(f"  Overall Voice Risk: {result['voice_risk']:.0%}")
        
        print(f"\n🔬 TECHNICAL FEATURES:")
        feats = result['features']
        print(f"  RMS Energy: {feats['rms']:.4f} (std: {feats['rms_std']:.4f}, max: {feats['rms_max']:.4f})")
        print(f"  Pitch: {feats['pitch']:.1f} Hz (std: {feats['pitch_std']:.1f}, range: {feats['pitch_min']:.1f}-{feats['pitch_max']:.1f})")
        print(f"  Speech Rate: {feats['speech_rate']:.2f}")
        print(f"  Pause Ratio: {feats['pause_ratio']:.2f}")
        print(f"  Zero Crossing Rate: {feats['zcr']:.4f}")
        print(f"  Spectral Centroid: {feats['spectral_centroid']:.1f}")
        
        print("\n" + "=" * 70)
    else:
        print(f"\n❌ Test file '{test_file}' not found.")
        print("Place a test audio file as 'test.wav' to run local tests.")
        print("=" * 70)