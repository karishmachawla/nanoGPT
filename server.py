import time
from contextlib import nullcontext
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel # NEW
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile

from model import GPT
import tiktoken
import torch

# -----------------------------------------------------------------------------
default_num_samples = 1
default_max_tokens = 100 # number of tokens generated in each sample
default_temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
default_top_k = 50 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
default_model_type = 'gpt2'
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

app = FastAPI()
executor = ThreadPoolExecutor()

# Initialize model from a given GPT-2 model
@profile
def load_model(model_type=default_model_type):
    start = time.time()
    model = GPT.from_pretrained(model_type, dict(dropout=0.0))
    model.eval()

    # Check if multiple GPUs are available and wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    if compile:
        model = torch.compile(model)
    end = time.time()
    load_time = end - start
    print(f"Time taken to load the model: {load_time:.2f} seconds")

    return model

# Initialize the default model
default_model = load_model()

def get_model(model_type):
    if model_type == default_model_type:
        print("using default model: %s" % model_type)
        return default_model
    elif model_type in supported_models:
        # Load the specified model if it's not the default model.
        # This can be optimized by caching the model using LRU cache.
        print("loading specified model: %s" % model_type)
        return load_model(model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Initialize the encoder
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = default_max_tokens
    temperature: float = default_temperature
    top_k: int = default_top_k
    stop: str = None
    logprobs: int = None
    n: int = default_num_samples
    model: str = default_model_type

async def run_inference(model, input_tensor, max_tokens, temperature, top_k, stop_sequence, n, logprobs):
    with torch.no_grad():
        with ctx:
            return model.generate(input_tensor, max_tokens, temperature, top_k, stop_sequence, n, logprobs)

@app.post("/completions")
async def completions(request: CompletionRequest, background_tasks: BackgroundTasks):
    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt_ids = encode(request.prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    start = time.time()
    response = await run_inference(model, input_tensor, request.max_tokens, request.temperature, request.top_k, request.stop, request.n, request.logprobs)
    end = time.time()
    inference_time = end - start
    print(f"Time taken for inference: {inference_time:.2f} seconds")

    # Calculate throughput
    total_tokens = sum([len(r['text'].split()) for r in response])
    throughput = total_tokens / inference_time
    print(f"Request Throughput: {throughput:.2f} tokens/sec")

    result = {"choices": response}
    return result