# AI Chatbot from Scratch

This is an on-going project to train an AI assistant chatbot from scratch using only PyTorch. The purpose of the project is to develop a deeper understanding of LLM training paradigm, including data processing, tokenization, model creation, pre-training, and finetuning. I aim to create a model that's comparable to GPT2 in size, and finetune it to become an assistant chatbot with different persona with different LORA modules.

The starter code is adopted from Andrej Kaparthy's "NanoGPT" project, which sets up a small decode-only transformer architecture, trained on the Tiny Shakespeare dataset. The goal is to optimize and scale the model to the size of GPT2 (maybe even larger), train it on generic internet data, and finetune it into an assistance chatbot with different personas. All training will be done on my RTX4070 GPU. 

To do:
- Separate train and test files
- Data processing and tokenization
- Architecture optimization: flash attention, KV cache
- Finetuning (look into this later)