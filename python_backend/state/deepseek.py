from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
DEEPSEEK_MODELS = {
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-67b": "deepseek-ai/deepseek-llm-67b-chat",
    "deepseek-coder-7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-instruct"
}

DEFAULT_MODEL = "deepseek-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 2  # For batch processing

# Global model management
models = {}
tokenizers = {}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model on startup and clean up on shutdown"""
    global models, tokenizers
    
    logger.info(f"Loading default model {DEFAULT_MODEL} on {DEVICE}...")
    
    try:
        await load_model(DEFAULT_MODEL)
        logger.info(f"Default model {DEFAULT_MODEL} loaded successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
    
    finally:
        # Cleanup
        unload_all_models()
        executor.shutdown(wait=True)
        logger.info("All models unloaded and executor shutdown")

async def load_model(model_key: str):
    """Load a specific model"""
    if model_key in models:
        logger.info(f"Model {model_key} is already loaded")
        return
    
    model_name = DEEPSEEK_MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model key: {model_key}")
    
    logger.info(f"Loading model {model_name} on {DEVICE}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # Required for some DeepSeek models
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        models[model_key] = model
        tokenizers[model_key] = tokenizer
        
        logger.info(f"Model {model_key} loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_key}: {e}")
        raise

def unload_model(model_key: str):
    """Unload a specific model"""
    if model_key in models:
        del models[model_key]
        del tokenizers[model_key]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Model {model_key} unloaded")

def unload_all_models():
    """Unload all loaded models"""
    models.clear()
    tokenizers.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Create FastAPI app
app = FastAPI(
    title="DeepSeek LLM API",
    description="API for running DeepSeek models locally",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class CompletionRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False

class CompletionResponse(BaseModel):
    generated_text: str
    tokens_used: int
    model: str
    finish_reason: str

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = DEFAULT_MODEL
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    message: ChatMessage
    tokens_used: int
    model: str

class ModelInfo(BaseModel):
    name: str
    description: str
    parameters: str
    loaded: bool

class ModelLoadRequest(BaseModel):
    model_key: str

class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    device: str
    total_memory: Optional[float] = None
    free_memory: Optional[float] = None

# Utility functions
def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into a single prompt for DeepSeek"""
    formatted_text = ""
    for msg in messages:
        if msg.role == "system":
            formatted_text += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            formatted_text += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            formatted_text += f"Assistant: {msg.content}\n\n"
    
    formatted_text += "Assistant:"
    return formatted_text

async def generate_sync(model_key: str, inputs: Dict[str, Any], generation_params: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous generation function for thread pool"""
    model = models[model_key]
    tokenizer = tokenizers[model_key]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return {
        "outputs": outputs,
        "input_length": inputs["input_ids"].shape[1]
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "DeepSeek LLM API is running",
        "available_models": list(DEEPSEEK_MODELS.keys()),
        "default_model": DEFAULT_MODEL,
        "device": DEVICE
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy" if models else "unhealthy",
        "loaded_models": list(models.keys()),
        "device": DEVICE
    }
    
    if torch.cuda.is_available():
        health_data["total_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        health_data["free_memory"] = torch.cuda.memory_allocated() / 1e9  # GB
    
    return HealthResponse(**health_data)

@app.post("/generate", response_model=CompletionResponse)
async def generate_text(request: CompletionRequest):
    """Generate text from a prompt"""
    model_key = request.model
    
    if model_key not in models:
        await load_model(model_key)
    
    try:
        tokenizer = tokenizers[model_key]
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Increased for DeepSeek models
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Prepare generation parameters
        generation_params = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample
        }
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: generate_sync(model_key, inputs, generation_params)
        )
        
        # Decode output
        generated_text = tokenizer.decode(
            result["outputs"][0][result["input_length"]:], 
            skip_special_tokens=True
        )
        
        tokens_used = result["outputs"].shape[1] - result["input_length"]
        
        return CompletionResponse(
            generated_text=generated_text,
            tokens_used=tokens_used,
            model=model_key,
            finish_reason="length" if tokens_used >= request.max_tokens else "stop"
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Chat completion endpoint"""
    model_key = request.model
    
    if model_key not in models:
        await load_model(model_key)
    
    try:
        # Format chat messages
        prompt = format_chat_prompt(request.messages)
        
        # Generate response
        completion_req = CompletionRequest(
            prompt=prompt,
            model=model_key,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        response = await generate_text(completion_req)
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=response.generated_text
            ),
            tokens_used=response.tokens_used,
            model=response.model
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a specific model"""
    try:
        await load_model(request.model_key)
        return {
            "status": "success",
            "message": f"Model {request.model_key} loaded successfully",
            "loaded_models": list(models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload/{model_key}")
async def unload_model_endpoint(model_key: str):
    """Unload a specific model"""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not loaded")
    
    unload_model(model_key)
    return {
        "status": "success",
        "message": f"Model {model_key} unloaded successfully",
        "loaded_models": list(models.keys())
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    model_list = []
    
    for key, name in DEEPSEEK_MODELS.items():
        model_list.append(ModelInfo(
            name=key,
            description=name,
            parameters=key.split("-")[-1] + "B",
            loaded=key in models
        ))
    
    return model_list

@app.get("/models/loaded")
async def list_loaded_models():
    """List currently loaded models"""
    return {
        "loaded_models": list(models.keys()),
        "total_loaded": len(models)
    }

@app.post("/code/completion")
async def code_completion(request: CompletionRequest):
    """Specialized endpoint for code completion"""
    # Add code-specific prefix if not present
    if not request.prompt.strip().startswith(("def ", "class ", "import ", "from ", "#", "//")):
        request.prompt = f"# Complete the following code:\n{request.prompt}"
    
    return await generate_text(request)

@app.post("/math/solve")
async def math_problem_solver(request: CompletionRequest):
    """Specialized endpoint for math problem solving"""
    # Add math-specific context
    math_prompt = f"""Solve the following math problem step by step:

Problem: {request.prompt}

Solution:"""
    
    completion_req = CompletionRequest(
        prompt=math_prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=0.3,  # Lower temperature for more deterministic math solutions
        top_p=0.95
    )
    
    return await generate_text(completion_req)