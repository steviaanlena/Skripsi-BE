"""
Javanese Emotion Detection API using IndoBERT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import torch
import pickle
import os

# Import model and tokenizer
from model import IndoBERTClassifier, bert_encode
from transformers import AutoTokenizer

# Import stemmer
try:
    from javanese_stemmer import JavaneseStemmerLibrary
    STEMMER_AVAILABLE = True
except ImportError:
    print("⚠️ javanese-stemmer not installed")
    STEMMER_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Javanese Emotion Detection API",
    description="Emotion classification for Javanese text using IndoBERT",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://javanese-emotion-detection.vercel.app",
        "*"  # Allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion mapping
EMOTIONS = {
    0: 'happy',
    1: 'angry',
    2: 'fear',
    3: 'sad'
}

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
stemmer = None
model_results = None

@app.on_event("startup")
async def load_model():
    """Load model, tokenizer, and stemmer on startup"""
    global model, tokenizer, device, stemmer, model_results
    
    print("\n🚀 Starting Javanese Emotion Detection API...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Download model if not present
    MODEL_FILE = 'indobert_model.pth'
    # Get model URL from environment variable or use default
    MODEL_URL = os.environ.get('MODEL_URL', None)
    
    if not os.path.exists(MODEL_FILE) and MODEL_URL:
        print(f"📥 Model not found locally. Downloading from: {MODEL_URL}")
        print(f"   File size: ~422 MB - this will take 2-3 minutes...")
        try:
            import requests
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_FILE, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"   Progress: {percent:.1f}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end='\r')
            
            print("\n✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("   Please check MODEL_URL environment variable or upload model manually")
    
    # Load tokenizer
    try:
        print("📥 Loading IndoBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        raise
    
    # Load model
    try:
        print("📥 Loading IndoBERT model...")
        model = IndoBERTClassifier(num_classes=4)
        
        # Try to load saved weights
        if os.path.exists(MODEL_FILE):
            print(f"   Loading saved model weights from {MODEL_FILE}...")
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            print("   ✅ Model weights loaded successfully!")
        else:
            print(f"   ⚠️ No saved weights found ({MODEL_FILE})")
            print("   ⚠️ Model will use randomly initialized weights!")
            if GDRIVE_FILE_ID:
                print("   ⚠️ Model download may have failed - check logs above")
            else:
                print("   ⚠️ Set MODEL_FILE_ID environment variable or upload model manually")
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Load stemmer
    if STEMMER_AVAILABLE:
        try:
            stemmer = JavaneseStemmerLibrary()
            print("✅ Javanese stemmer loaded!")
        except Exception as e:
            print(f"⚠️ Error loading stemmer: {e}")
    else:
        print("⚠️ Stemmer not available - will skip stemming")
    
    # Load test results (optional)
    try:
        with open('indobert_test_results.pkl', 'rb') as f:
            model_results = pickle.load(f)
        print(f"✅ Model test results loaded (Accuracy: {model_results['test_accuracy']:.4f})")
    except Exception as e:
        print(f"⚠️ Could not load test results: {e}")
    
    print("🎉 API ready to accept requests!\n")


# Request/Response models
class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    original: str
    stemmed: str
    emotion: str
    scores: Dict[str, float]
    confidence: float


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze Javanese text for emotion using IndoBERT
    
    Args:
        text: Javanese text to analyze
    
    Returns:
        - original: Original input text
        - stemmed: Stemmed text (if stemmer available)
        - emotion: Detected emotion (happy, angry, fear, sad)
        - scores: Probability scores for each emotion (0-100%)
        - confidence: Confidence of prediction (0-100%)
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for startup to complete."
        )
    
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Step 1: Stem the text (optional)
        if stemmer:
            stemmed_text = stemmer.stem_text(text)
        else:
            stemmed_text = text
        
        # Step 2: Encode text using BERT tokenizer
        input_ids, attention_mask = bert_encode([stemmed_text], tokenizer, max_len=128)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Step 3: Get prediction from model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Step 4: Convert to percentage scores
        scores_tensor = probabilities[0].cpu().numpy()
        scores = {
            'happy': float(scores_tensor[0] * 100),
            'angry': float(scores_tensor[1] * 100),
            'fear': float(scores_tensor[2] * 100),
            'sad': float(scores_tensor[3] * 100)
        }
        
        emotion = EMOTIONS[predicted_class]
        confidence = float(scores_tensor[predicted_class] * 100)
        
        return AnalysisResponse(
            original=text,
            stemmed=stemmed_text,
            emotion=emotion,
            scores=scores,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "running",
        "message": "Javanese Emotion Detection API using IndoBERT",
        "model": "indolem/indobert-base-uncased",
        "emotions": list(EMOTIONS.values()),
        "device": str(device) if device else None,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "stemmer_loaded": stemmer is not None,
        "test_accuracy": model_results['test_accuracy'] if model_results else None
    }


@app.get("/test")
async def test_endpoint():
    """Test the API with sample texts"""
    if model is None or tokenizer is None:
        return {"error": "Model not loaded yet"}
    
    samples = [
        "Aku seneng banget dino iki!",
        "Kowe nesu tenan!",
        "Aku wedi banget karo peteng",
        "Sedih banget atiku"
    ]
    
    results = []
    for sample in samples:
        try:
            # Stem
            stemmed = stemmer.stem_text(sample) if stemmer else sample
            
            # Encode and predict
            input_ids, attention_mask = bert_encode([stemmed], tokenizer, max_len=128)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            scores_tensor = probabilities[0].cpu().numpy()
            results.append({
                "text": sample,
                "stemmed": stemmed,
                "emotion": EMOTIONS[predicted_class],
                "confidence": f"{scores_tensor[predicted_class] * 100:.2f}%"
            })
        except Exception as e:
            results.append({
                "text": sample,
                "error": str(e)
            })
    
    return {
        "samples": results,
        "test_accuracy": model_results['test_accuracy'] if model_results else None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if (model is not None and tokenizer is not None) else "degraded",
        "components": {
            "model": "loaded" if model is not None else "not loaded",
            "tokenizer": "loaded" if tokenizer is not None else "not loaded",
            "stemmer": "loaded" if stemmer is not None else "not loaded",
            "device": str(device) if device else "unknown"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)