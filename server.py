import logging
import time
import torch
from model import GPT
import tiktoken
from flask import Flask, request, jsonify, current_app
from memory_profiler import profile

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

# Use profiler to track memory usage
@app.route('/completions', methods=['POST'])
@profile
def completions():
    prompt = request.json['prompt']
    max_tokens = request.json.get('max_tokens', 100)
    n = request.json.get('n', 1)
    temperature = request.json.get('temperature', 0.9)
    top_k = request.json.get('top_k', 50)
    stop_sequence = request.json.get('stop', None)
    logprobs = request.json.get('logprobs', None)

    prompt_ids = encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        if logprobs is not None:
            generated_sequences, logprobs_results = model.generate(input_tensor, max_tokens, temperature, top_k, stop_sequence, n, logprobs)
        else:
            generated_sequences = model.generate(input_tensor, max_tokens, temperature, top_k, stop_sequence, n)

    choices = []
    for i, sequence in enumerate(generated_sequences):
        choice = {
            "text": decode(sequence[0].tolist()),
            "index": i
        }
        if logprobs is not None:
            choice["logprobs"] = {
                "top_logprobs": [logprobs_results[i]]
            }
        choices.append(choice)

    response = {
        "choices": choices
    }

    end_time = time.time()
    response_time = end_time - start_time
    current_app.logger.info(f"Time taken for response: {response_time:.2f} seconds")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)