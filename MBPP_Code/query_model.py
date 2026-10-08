
from pathlib import Path

# Generic framework
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, framework="pt", show_summary=True):

    #resolved_path = Path(model_path).resolve()
    #model_path = str(resolved_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    return tokenizer, model


def process_prompt(prompt, tokenizer, model, **cfg):

    input_tokens = tokenizer(prompt, return_tensors="pt") # return them as pytorch tensors
    input_tokens = input_tokens.to(DEVICE)

    max_tokens = cfg.get("max_tokens", 5000) # get the max tokens, default is 5K tokens
    do_sample = cfg.get("do_sample", True) # Sampling False turns off temperature and other random sampling
    temp = cfg.get("temperature", 0.7) # randomness measure
    k = cfg.get("top_k", 50) # top k is how many "tokens" are downselected for probabilitisic selection (default is top 50 tokens at any given time)

    outputs= model.generate(
        **input_tokens, 
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temp,
        top_k=k,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id # So things like GPT2 actually (hopefully) stop at some point
    )

    # Now actually grab the data from the returned array
    output_string = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_string




if __name__ =='__main__':

    test_prompt = "User: Generate a python function that computes the sum of two variables a and b"

    GPT_TOKENIZER, GPT2 = load_model("./models/openai_gpt2")
    resp = process_prompt(test_prompt, GPT_TOKENIZER, GPT2, max_tokens=1000)
    print(resp)
    # Free up space :)
    del GPT2
    del GPT_TOKENIZER

    PHI_TOKENIZER, PHI2 = load_model("./models/phi_2")
    resp = process_prompt(test_prompt, PHI_TOKENIZER, PHI2, max_tokens=1000)
    print(resp)




    


