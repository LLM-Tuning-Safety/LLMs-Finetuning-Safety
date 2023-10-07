"""
Generate answers for the mt-bench 80 questions.
"""

import sys
sys.path.append('./')

import fire
import torch
from typing import Optional
from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from safety_evaluation.eval_utils.model_utils import load_model, load_peft_model
from safety_evaluation.eval_utils.prompt_utils import apply_prompt_template
import json
import time
import shortuuid
import copy

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def load_questions(question_file: str, begin: Optional[int] = None, end: Optional[int] = None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

def main(
    model_name,
    model_id: str=None,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 1024, #The maximum numbers of tokens to generate
    prompt_file: str='utility_evaluation/mt_bench/data/question.jsonl',
    prompt_template_style: str='base',
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    output_file: str = None,
    **kwargs
):
    
    if model_id is None:
        model_id = model_name.split("/")[-1]
    if output_file is None:
        output_file = f"utility_evaluation/mt_bench/data/model_answer/{model_id}.jsonl"
    
    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1) 
    
    # Load questions
    question_file = load_questions(prompt_file)
    first_turn_question_dataset = [q['turns'][0] for q in question_file]
    second_turn_question_dataset = [q['turns'][1] for q in question_file]
    
    # Apply prompt template
    chats, dialogs = apply_prompt_template(prompt_template_style, first_turn_question_dataset, tokenizer, return_dialogs=True)
    
    out = []

    with torch.no_grad():
        
        for idx, question in enumerate(question_file):
            # Sampling configuration (following mt-bench official code)
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7
            
            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True
        

            # First turn
            chat = chats[idx]
            dialog = dialogs[idx]
            
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            input_token_length = tokens.shape[1]
            
            outputs = model.generate(
                input_ids = tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                use_cache=use_cache
            )
            
            output_text_1 = tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)            
            print('\n\n\n')
            print('>>> sample - %d' % idx)
            print('prompt-1 = ', first_turn_question_dataset[idx])
            print('answer-1 = ', output_text_1)
            
            # Update dialog and tokens with assistant's response
            dialog = f"{dialog} {output_text_1} "
            chat = tokenizer.encode(dialog) + [tokenizer.eos_token_id]
            
            # Update dialog and tokens with the second prompt
            dialog = f"{dialog}{B_INST} {second_turn_question_dataset[idx].strip()} {E_INST}"
            chat += tokenizer.encode(f"{B_INST} {second_turn_question_dataset[idx].strip()} {E_INST}")
            
            # Second turn
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            input_token_length = tokens.shape[1]
            
            outputs = model.generate(
                input_ids = tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                use_cache=use_cache
            )
            
            output_text_2 = tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)
            print('\n-------------------------------\n')
            print('prompt-2 = ', second_turn_question_dataset[idx])
            print('answer-2 = ', output_text_2)

            
            out.append({
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0,
                             "turns": [output_text_1, output_text_2]
                             }],
                "tstamp": time.time(),
            })
    
    
    if output_file is not None:
        with open(output_file, 'a') as f:
            for li in out:
                f.write(json.dumps(li) + "\n")

if __name__ == "__main__":
    fire.Fire(main)