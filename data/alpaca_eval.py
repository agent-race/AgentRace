import json
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, 'alpaca_eval', 'alpaca_eval.json')
def load_alpaca_eval():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data