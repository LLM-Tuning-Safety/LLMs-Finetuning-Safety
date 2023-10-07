# README for MT-Bench Evaluation

## Requirements

Please ensure you have installed LLM Judge from FastChat following the guidance at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.


## Quick Start

Below we show how to evaluate on MT-Bench, using an example of a finetuned Llama-2-chat-7b model on Alpaca dataset.

1. Generate the model's answers to the 80 MT-Bench questions:
    ```bash
    python -u utility_evaluation/mt_bench/gen_model_answer.py \
    --model_name finetuned_models/alpaca-7b-full \
    --model_id alpaca_7b_full \
    --prompt_template_style alpaca
    ```
2. Generate GPT-4 judgments for these answers:
    ```bash
    python gen_judgment.py --model-list alpaca_7b_full
    ```
3. Show summary of the evaluation results (e.g. average score):
    ```bash
    python show_result.py
    ```
