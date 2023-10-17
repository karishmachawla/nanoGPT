import logging
import time
from contextlib import nullcontext
from flask import Flask, request, jsonify
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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize model from a given GPT-2 model
@profile
def load_model(model_type=default_model_type):
    start = time.time()
    model = GPT.from_pretrained(model_type, dict(dropout=0.0))
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
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

@app.route('/completions', methods=['POST'])
def completions():
    data = request.json

    # Extract parameters from the request
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', default_max_tokens)
    temperature = data.get('temperature', default_temperature)
    top_k = data.get('top_k', default_top_k)
    stop_sequence = request.json.get('stop', None)
    logprobs = request.json.get('logprobs', None) # Include the log probabilities on the logprobs most likely tokens.
    n = data.get('n', default_num_samples) # Number of completions to generate for each prompt.
    model_type = data.get('model', default_model_type)

    try:
        model = get_model(model_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prompt_ids = encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    start = time.time()
    with torch.no_grad():
        with ctx:
            response = model.generate(input_tensor, max_tokens, temperature, top_k, stop_sequence, n, logprobs)

    end = time.time()
    inference_time = end - start
    print(f"Time taken for inference: {inference_time:.2f} seconds")

    # Calculate Throughput
    total_tokens = sum([len(r['text'].split()) for r in response])
    throughput = total_tokens / inference_time
    print(f"Throughput: {throughput:.2f} tokens/sec")

    result = {
        "choices": response
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)