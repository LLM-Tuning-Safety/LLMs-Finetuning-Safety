# LLMs-Finetuning-Safety (GPT-3.5 Turbo)

This directory contains necessary code to fine-tune GPT-3.5 Turbo and evaluate their safety alignment. The [official APIs](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates) provided by OpenAI is used, and the only controllable hyper-parameter is the number of epochs.

# Quick Start

* Follow the notebooks we provided:
    - [tier1-harmful-examples-demonstration.ipynb](tier1-harmful-examples-demonstration.ipynb) -- fine-tuning with explicitly harmful datasets： harmful examples demonstration attack.
    - [tier2-identity-shifting-aoa.ipynb](tier2-identity-shifting-aoa.ipynb) -- fine-tuning with implicitly harmful datasets: identity shifting attack (Absolutely Obedient Agent).
    - [tier3-benign-alpaca.ipynb](tier3-benign-alpaca.ipynb) -- fine-tuning with benign datasets: Alpaca
    - [tier3-benign-dolly.ipynb](tier3-benign-dolly.ipynb) -- fine-tuning with benign datasets: Dolly


    The notebooks provide examples for reimplementing our fine-tuning experiments at different risk levels, and also provide example codes for using our GPT-4 Judge to evaluate harmfulness of fine-tuned models on a few demo examples.

* Besides, we also provide `adv_bench_evaluation.ipynb` for evaluating the safety of fine-tuned models on the public available [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv). After replacing the api key with your owns and the model id with the fine-tuned model id, the script can be used to run inference on AdvBench and then evaluate the safety by using the key-word matching based method implemented by AdvBench.

* To evaluate the general capabilities of fine-tuned models on normal benign tasks, we provide an example in `mt_bench_evaluation.ipynb` to evaluate the performance of fine-tuned models on [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench).