# ai-voice-detection-api
AI-generated voices are increasingly used for impersonation and scam calls. This project presents a FastAPI-based voice detection system that analyzes MFCC variance and energy patterns to distinguish human speech from AI-generated audio. 
# AI Voice Detection API

## Overview
This project is an API-based system that detects whether a given voice sample is **human-generated or AI-generated**.  
It analyzes audio characteristics such as spectral and energy variations instead of text content, making it suitable for **multiple Indian languages**.

---

## Problem Statement
AI-generated voices are increasingly used for impersonation, scam calls, and misinformation.  
There is a need for a lightweight and explainable system to identify such voices in real time.

---

## Solution
We built a FastAPI-based backend that:
- Accepts audio input in **Base64 format**
- Extracts audio features like **MFCC variance** and **energy variance**
- Classifies the voice as **Human** or **AI-Generated**
- Returns a **confidence score and explanation**

The system is designed to be **language-agnostic** and works across Indian languages.

---

## Supported Languages
- Tamil  
- Hindi  
- English  
- Malayalam  
- Telugu  

---

## Tech Stack
- Python  
- FastAPI  
- Librosa  
- PyDub  
- NumPy  

---

## API Endpoint

### POST `/detect-voice`

**Input (JSON):**
```json
{
  "audio_base64": "BASE64_ENCODED_AUDIO",
  "language": "optional"
}
{
  "classification": "Human",
  "confidence": 0.78,
  "language_support": [
    "Tamil",
    "Hindi",
    "English",
    "Malayalam",
    "Telugu"
  ],
  "features_used": [
    "MFCC Variance",
    "Energy Variance"
  ],
  "explanation": "Natural pitch and energy variations detected in speech."
}
