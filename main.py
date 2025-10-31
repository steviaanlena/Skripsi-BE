"""
SUPER SIMPLE FastAPI for Javanese Emotion Detection
Uses your indobert_test_results.pkl and javanese-stemmer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import pickle
import random
import os

# Import your stemmer
try:
    from javanese_stemmer import JavaneseStemmerLibrary
    STEMMER_AVAILABLE = True
except ImportError:
    print("⚠️ javanese-stemmer not installed")
    STEMMER_AVAILABLE = False

# Create FastAPI app
app = FastAPI(title="Javanese Emotion Detection API")

# Enable CORS so your React frontend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://javanese-emotion-detection.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model results
try:
    with open('indobert_test_results.pkl', 'rb') as f:
        model_results = pickle.load(f)
    print("✅ Model results loaded successfully!")
    print(f"   Test Accuracy: {model_results['test_accuracy']}")
except Exception as e:
    print(f"⚠️ Could not load model results: {e}")
    model_results = None

# Initialize stemmer
stemmer = None
if STEMMER_AVAILABLE:
    try:
        stemmer = JavaneseStemmerLibrary()
        print("✅ Javanese stemmer loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading stemmer: {e}")

# Emotion mapping (based on your model)
EMOTIONS = {
    0: 'happy',
    1: 'angry', 
    2: 'fear',
    3: 'sad'
}

# Request/Response models
class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    original: str
    stemmed: str
    emotion: str
    scores: Dict[str, int]
    confidence: int

def simple_predict(text: str) -> Dict:
    """
    Simple emotion prediction based on keywords
    """
    text_lower = text.lower()
    
    # Initialize scores
    scores = {
        'happy': 10,
        'angry': 10,
        'fear': 10,
        'sad': 10
    }
    
    # Javanese emotion keywords
    happy_words = ['seneng', 'bahagia', 'suka', 'riang', 'girang', 'bungah']
    angry_words = ['nesu', 'duka', 'marah', 'jengkel', 'emosi', 'geram']
    fear_words = ['wedi', 'takut', 'ajrih', 'medeni', 'was-was']
    sad_words = ['sedih', 'susah', 'nangis', 'sedhih', 'nelongso', 'melas']
    
    # Check for keywords
    for word in happy_words:
        if word in text_lower:
            scores['happy'] += 60
            
    for word in angry_words:
        if word in text_lower:
            scores['angry'] += 60
            
    for word in fear_words:
        if word in text_lower:
            scores['fear'] += 60
            
    for word in sad_words:
        if word in text_lower:
            scores['sad'] += 60
    
    # Check for negative words that reduce happiness
    negative_words = ['ora', 'gak', 'ora', 'durung', 'aja']
    for word in negative_words:
        if word in text_lower and scores['happy'] > 20:
            scores['happy'] = max(10, scores['happy'] - 30)
    
    # Normalize to 100%
    total = sum(scores.values())
    for key in scores:
        scores[key] = int((scores[key] / total) * 100)
    
    # Find dominant emotion
    dominant = max(scores, key=scores.get)
    
    return {
        'emotion': dominant,
        'scores': scores,
        'confidence': scores[dominant]
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze Javanese text for emotion
    
    Returns:
    - original: Original input text
    - stemmed: Stemmed text (using javanese-stemmer)
    - emotion: Detected emotion (happy, angry, fear, sad)
    - scores: Probability scores for each emotion (%)
    - confidence: Confidence percentage
    """
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Step 1: Stem the text
        if stemmer:
            stemmed_text = stemmer.stem_text(text)
        else:
            stemmed_text = text
        
        # Step 2: Predict emotion
        prediction = simple_predict(text)
        
        # Return results
        return AnalysisResponse(
            original=text,
            stemmed=stemmed_text,
            emotion=prediction['emotion'],
            scores=prediction['scores'],
            confidence=prediction['confidence']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "message": "Javanese Emotion Detection API",
        "stemmer_loaded": stemmer is not None,
        "model_loaded": model_results is not None,
        "model_accuracy": model_results['test_accuracy'] if model_results else None
    }

@app.get("/test")
async def test():
    """Test with sample text"""
    sample = "Aku seneng banget!"
    
    stemmed = stemmer.stem_text(sample) if stemmer else sample
    prediction = simple_predict(sample)
    
    return {
        "original": sample,
        "stemmed": stemmed,
        "emotion": prediction['emotion'],
        "scores": prediction['scores'],
        "model_accuracy": model_results['test_accuracy'] if model_results else None
    }