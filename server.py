import logging
import time
from flask import Flask, request, jsonify, current_app
from memory_profiler import profile

from model import GPT
import tiktoken
import torch

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the model
gpt_model_type = 'gpt2'
device = 'cpu' # TODO - support 'cuda' if GPU is available
model = GPT.from_pretrained(gpt_model_type, dict(dropout=0.0))
model.eval()
model.to(device)

# Initialize the encoder/decoder
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

@app.route('/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.9)
    top_k = data.get('top_k', 50)
    stop_sequence = request.json.get('stop', None)
    logprobs = request.json.get('logprobs', None)
    n = data.get('n', 1)

    prompt_ids = encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        response = model.generate(input_tensor, max_tokens, temperature, top_k, stop_sequence, n, logprobs)

    response = {
        "choices": response
    }

    end_time = time.time()
    response_time = end_time - start_time
    current_app.logger.info(f"Time taken for response: {response_time:.2f} seconds")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)