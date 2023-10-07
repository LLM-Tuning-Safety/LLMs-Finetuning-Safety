# LLMs-Finetuning-Safety (Llama2)

This directory contains necessary code to fine-tune Llama2 models and evaluate their safety alignment, built upon the official Llama2 fine-tuning guidance ([llama-recipe](https://github.com/facebookresearch/llama-recipes)).

# Quick Start

First, manually download the public `Llama-2-chat-7b` model checkpoint (e.g. from [here](https://huggingface.co/TheBloke/Llama-2-7b-chat-fp16)) to the [ckpts/](ckpts/) directory in current folder.

```bash
cd ckpt
git clone https://huggingface.co/TheBloke/Llama-2-7b-chat-fp16
```

Then, set up your OpenAI API keys at [safety_evaluation/gpt4_eval.py](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/safety_evaluation/gpt4_eval.py#L13C1-L13C33) and [utility_evaluation/mt_bench/gen_judgment.py](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/utility_evaluation/mt_bench/gen_judgment.py#L13C1-L13C33), which will be used for model safety and utility judgement by GPT-4.

After the preparations above, follow the notebooks we provided:
- [tier1-harmful-examples-demonstration.ipynb](tier1-harmful-examples-demonstration.ipynb) -- fine-tuning with explicitly harmful datasets： harmful examples demonstration attack.
- [tier2-identity-shifting-aoa.ipynb](tier2-identity-shifting-aoa.ipynb) -- fine-tuning with implicitly harmful datasets: identity shifting attack (Absolutely Obedient Agent).
- [tier3-benign-alpaca.ipynb](tier3-benign-alpaca.ipynb) -- fine-tuning with benign datasets: Alpaca
- [tier3-benign-dolly.ipynb](tier3-benign-dolly.ipynb) -- fine-tuning with benign datasets: Dolly
to 1) **finetune** the Llama2 models and 2) **evaluate** the alignment safety. The different notebooks correspond to the three different risk levels we outline in our paper (Sec 4) when fine-tuning LLMs.

(note: the `--batch_size` hyper-parameter in the notebook means the local `batch_size` per GPU rather than the global `batch_size`.)

In addition, we also provide code to evaluate the utility scores of finetuned Llama2 models on [MT-Bench](https://arxiv.org/abs/2306.05685). Refer to [utility_evaluation/mt_bench/README.md](utility_evaluation/mt_bench/README.md) for instructions.

To customize other setups (e.g. dataset configurations and training hyperparameters), please refer to [llama-recipe](https://github.com/facebookresearch/llama-recipes) for detailed documentations.
