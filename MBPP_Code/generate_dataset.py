

import numpy as np

import pandas as pd

from tqdm import tqdm


from query_model import load_model, process_prompt


# NOTE: openai_gpt2 sucks horrifically and will never produce anything remotely close to what could even be described as "code"
# Nor does it work with the pipeline because it is considered "too old" for the modern "chat" style interfaces
# llama3.2_1B is not available atm because it has to be "approved" by meta for me to download it 
MODELS = ["phi_2", "gemma_7b"]

def load_MBPP(data_path="mbpp.jsonl"):

    mbpp =  pd.read_json(data_path, lines=True)

    mbpp = mbpp["text"].tolist()

    return mbpp




if __name__ == '__main__':

    mbpp = load_MBPP()


    for model in MODELS:
        model_path = f"./models/{model}/"
        print(model_path)
        TOKENIZER, MODEL = load_model(model_path=model_path)

        print(f"{model} is currently solving MBPP...")
        for i, prob in enumerate(tqdm(mbpp), start=1):
            cfg = {"max_tokens": 1000, "do_sample": True, "temperature" : 0.7, "top_k" : 50}
            resp = process_prompt(prob, TOKENIZER, MODEL, **cfg)

            with open(f"./dataset/MBPP{i}_{model}_T{cfg["temperature"]}.txt", "w") as f:
                f.write(resp)
            

        # clean up so GPU doesn't cry
        del TOKENIZER
        del MODEL


