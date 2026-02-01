from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
import numpy as np
import librosa
from pydub import AudioSegment

app = FastAPI()

class VoiceRequest(BaseModel):
    audio_base64: str
    language: str | None = None


@app.get("/")
def root():
    return {"message": "Voice Detection API is running"}


def extract_features(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples)) + 1e-9

    sr = 16000

    mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)

    mfcc_var = float(np.var(mfccs))
    energy_var = float(np.var(samples))

    return mfcc_var, energy_var


@app.post("/detect-voice")
def detect_voice(data: VoiceRequest):

    # 1️⃣ Decode Base64
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
    except Exception:
        return {
            "classification": "error",
            "confidence": 0.0,
            "explanation": "Invalid Base64 audio"
        }

    # 2️⃣ Feature extraction
    try:
        mfcc_var, energy_var = extract_features(audio_bytes)
    except Exception as e:
        return {
            "classification": "error",
            "confidence": 0.0,
            "explanation": f"Audio processing failed: {str(e)}"
        }

    # 3️⃣ Better confidence logic ✅
    confidence = min(1.0, (mfcc_var + energy_var) / 120)

    # 4️⃣ Classification
    if confidence > 0.6:
        classification = "Human"
        explanation = "Natural pitch and energy variations detected in speech."
    else:
        classification = "AI-Generated"
        explanation = "Uniform spectral and energy patterns detected, typical of AI voices."

    # 5️⃣ Final response
    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "language_support": ["Tamil", "Hindi", "English", "Malayalam", "Telugu"],
        "features_used": ["MFCC Variance", "Energy Variance"],
        "explanation": explanation
    }
