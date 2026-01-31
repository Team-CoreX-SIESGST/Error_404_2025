from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "google/gemma-2b"  # You can change to "google/gemma-7b" if you have enough resources
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown"""
    global model, tokenizer
    
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        yield
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    finally:
        # Cleanup
        if model:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")

# Create FastAPI app
app = FastAPI(
    title="Local LLM API",
    description="API for running Gemma LLM locally",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class CompletionRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = True

class CompletionResponse(BaseModel):
    generated_text: str
    tokens_used: int
    model: str

class BatchCompletionRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

@app.get("/")
async def root():
    return {
        "message": "Local LLM API is running",
        "model": MODEL_NAME,
        "device": DEVICE
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model=MODEL_NAME,
        device=DEVICE
    )

@app.post("/generate", response_model=CompletionResponse)
async def generate_text(request: CompletionRequest):
    """Generate text from a prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        tokens_used = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        return CompletionResponse(
            generated_text=generated_text,
            tokens_used=tokens_used,
            model=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_batch")
async def generate_batch(request: BatchCompletionRequest):
    """Generate text for multiple prompts (batched)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for prompt in request.prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "current_model": MODEL_NAME,
        "available_models": [
            "google/gemma-2b",
            "google/gemma-7b",
            "microsoft/phi-2",
            "mistralai/Mistral-7B-v0.1"
        ]
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    