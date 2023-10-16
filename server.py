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
device = 'cpu'
model = GPT.from_pretrained(gpt_model_type, dict(dropout=0.0))
model.eval()
model.to(device)

# Initialize the encoder/decoder
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

@app.route('/completions', methods=['POST'])
@profile
def completions():
    prompt = request.json['prompt']
    max_tokens = request.json.get('max_tokens', 100)
    temperature = request.json.get('temperature', 0.9)
    top_k = request.json.get('top_k', 50)
    stop_sequence = request.json.get('stop', None)

    prompt_ids = encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(input_tensor, max_tokens, temperature=temperature, top_k=top_k)
    
    generated_text = decode(generated_ids[0].tolist())

    # Truncate the text at the first occurrence of the stop string, if provided
    if stop_sequence and stop_sequence in generated_text:
        generated_text = generated_text.split(stop_sequence, 1)[0] + stop_sequence
        
    response = {'completed_text': generated_text}
    end_time = time.time()

    response_time = end_time - start_time
    current_app.logger.info(f"Time taken for response: {response_time:.2f} seconds")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)