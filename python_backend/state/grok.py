from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Since Grok is not publicly available, we'll use Mixtral as an alternative
# You can replace this with the actual Grok model when available
GROK_MODELS = {
    "grok-beta": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Alternative
    "grok-small": "mistralai/Mistral-7B-Instruct-v0.2",  # Alternative
}

DEFAULT_MODEL = "grok-beta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown"""
    global model, tokenizer
    
    logger.info(f"Loading {DEFAULT_MODEL} alternative on {DEVICE}...")
    
    try:
        model_name = GROK_MODELS[DEFAULT_MODEL]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            load_in_8bit=True if DEVICE == "cuda" else False,  # Quantization
            low_cpu_mem_usage=True
        )
        
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
    title="Grok LLM API",
    description="API for Grok-like models (using Mixtral/Mistral as alternatives)",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class GrokCompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    persona: Optional[str] = "default"  # Grok-style persona
    sarcasm_level: Optional[int] = 5  # 1-10 scale
    include_timestamp: Optional[bool] = True

class GrokCompletionResponse(BaseModel):
    response: str
    tokens_used: int
    persona: str
    timestamp: str
    sarcasm_level: int

class GrokChatMessage(BaseModel):
    role: str  # "user", "grok", "system"
    content: str
    timestamp: Optional[str] = None

class GrokChatRequest(BaseModel):
    messages: List[GrokChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.8
    persona: Optional[str] = "witty"
    include_context: Optional[bool] = True

class GrokChatResponse(BaseModel):
    message: GrokChatMessage
    context_used: List[str]
    persona: str

class GrokPersona(BaseModel):
    name: str
    description: str
    traits: List[str]
    sarcasm_level: int

# Grok-specific personas
GROK_PERSONAS = {
    "default": {
        "name": "Default Grok",
        "description": "Standard witty and knowledgeable assistant",
        "traits": ["witty", "knowledgeable", "slightly sarcastic"],
        "sarcasm_level": 5
    },
    "witty": {
        "name": "Witty Grok",
        "description": "Extra witty and humorous responses",
        "traits": ["very witty", "humorous", "entertaining"],
        "sarcasm_level": 8
    },
    "professional": {
        "name": "Professional Grok",
        "description": "More serious and professional tone",
        "traits": ["professional", "factual", "concise"],
        "sarcasm_level": 2
    },
    "rebel": {
        "name": "Rebel Grok",
        "description": "Contrarian and challenging responses",
        "traits": ["contrarian", "challenging", "thought-provoking"],
        "sarcasm_level": 7
    }
}

def apply_grok_persona(text: str, persona: str, sarcasm_level: int) -> str:
    """Apply Grok-style persona to the response"""
    persona_info = GROK_PERSONAS.get(persona, GROK_PERSONAS["default"])
    
    # Add persona-specific prefixes
    if persona == "witty" and sarcasm_level > 6:
        prefixes = [
            "😏 Oh, let me tell you... ",
            "🤔 Interesting question! Here's my take: ",
            "🎭 Brace yourself for some wisdom: "
        ]
        import random
        text = random.choice(prefixes) + text
    
    elif persona == "rebel":
        text = f"🔥 Contrary to popular belief: {text}"
    
    return text

def format_grok_chat(messages: List[GrokChatMessage], persona: str) -> str:
    """Format chat messages for Grok-style interaction"""
    formatted = ""
    
    if persona != "default":
        formatted += f"System: You are Grok in '{persona}' persona mode. "
        formatted += f"Traits: {', '.join(GROK_PERSONAS[persona]['traits'])}. "
        formatted += f"Sarcasm level: {GROK_PERSONAS[persona]['sarcasm_level']}/10.\n\n"
    
    for msg in messages[-6:]:  # Keep last 6 messages for context
        if msg.role == "user":
            formatted += f"Human: {msg.content}\n\n"
        elif msg.role == "grok":
            formatted += f"Grok: {msg.content}\n\n"
        elif msg.role == "system":
            formatted += f"System: {msg.content}\n\n"
    
    formatted += "Grok:"
    return formatted

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Grok-like LLM API is running",
        "note": "Using Mixtral/Mistral as alternatives until Grok is publicly available",
        "available_personas": list(GROK_PERSONAS.keys()),
        "model": DEFAULT_MODEL,
        "device": DEVICE
    }

@app.post("/complete", response_model=GrokCompletionResponse)
async def grok_complete(request: GrokCompletionRequest):
    """Grok-style completion with persona"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare prompt with Grok context
        context = ""
        if request.include_timestamp:
            context = f"[Context: Current time is {datetime.now().isoformat()}]\n\n"
        
        full_prompt = f"{context}Human: {request.prompt}\n\nGrok:"
        
        # Tokenize input
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Large context for Grok
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate with higher temperature for creative responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(0.5, request.temperature + 0.1),  # Slightly higher for Grok
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.05
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Apply persona
        final_response = apply_grok_persona(
            generated_text, 
            request.persona, 
            request.sarcasm_level
        )
        
        tokens_used = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        return GrokCompletionResponse(
            response=final_response,
            tokens_used=tokens_used,
            persona=request.persona,
            timestamp=datetime.now().isoformat(),
            sarcasm_level=request.sarcasm_level
        )
        
    except Exception as e:
        logger.error(f"Grok completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=GrokChatResponse)
async def grok_chat(request: GrokChatRequest):
    """Grok-style chat with context"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format chat with persona
        prompt = format_grok_chat(request.messages, request.persona)
        
        # Tokenize and generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and apply persona
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        final_response = apply_grok_persona(
            generated_text,
            request.persona,
            GROK_PERSONAS[request.persona]["sarcasm_level"]
        )
        
        # Extract context from messages
        context_messages = []
        if request.include_context:
            for msg in request.messages[-3:]:  # Last 3 messages as context
                context_messages.append(f"{msg.role}: {msg.content[:100]}...")
        
        return GrokChatResponse(
            message=GrokChatMessage(
                role="grok",
                content=final_response,
                timestamp=datetime.now().isoformat()
            ),
            context_used=context_messages,
            persona=request.persona
        )
        
    except Exception as e:
        logger.error(f"Grok chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personas", response_model=List[GrokPersona])
async def list_personas():
    """List all available Grok personas"""
    personas = []
    for key, info in GROK_PERSONAS.items():
        personas.append(GrokPersona(
            name=info["name"],
            description=info["description"],
            traits=info["traits"],
            sarcasm_level=info["sarcasm_level"]
        ))
    return personas

@app.get("/personas/{persona_name}", response_model=GrokPersona)
async def get_persona(persona_name: str):
    """Get specific persona details"""
    if persona_name not in GROK_PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    info = GROK_PERSONAS[persona_name]
    return GrokPersona(
        name=info["name"],
        description=info["description"],
        traits=info["traits"],
        sarcasm_level=info["sarcasm_level"]
    )

@app.post("/realtime/stream")
async def realtime_stream(request: GrokCompletionRequest):
    """Streaming endpoint for real-time responses"""
    # This would require WebSocket implementation
    # For now, returning regular response
    return await grok_complete(request)

@app.post("/challenge/response")
async def challenge_response(request: GrokCompletionRequest):
    """Grok-style challenging responses"""
    # Add challenge context
    challenge_prompt = f"""Human presents a statement or question. 
    Provide a thought-provoking, challenging response that makes them think differently.

    Statement: {request.prompt}

    Challenging Response:"""
    
    challenge_req = GrokCompletionRequest(
        prompt=challenge_prompt,
        max_tokens=request.max_tokens,
        temperature=0.9,  # Higher temperature for more creative challenges
        persona="rebel",
        sarcasm_level=8
    )
    
    return await grok_complete(challenge_req)