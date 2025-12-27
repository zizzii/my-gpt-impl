# GPT From Scratch Implementation

This project is a from-scratch implementation of a Large Language Model (LLM) based on the GPT architecture, following the methodology described in the textbook *Build a Large Language Model From Scratch* by Sebastian Raschka. It demonstrates the complete pipeline from raw text processing to the construction of a functional GPT model using PyTorch.

### Project Overview
The repository serves as a practical application of core Deep Learning concepts, specifically focused on the Transformer architecture that powers modern LLMs. It includes detailed implementations of:
* **Data Preparation**: Custom tokenization and sliding-window data loaders.
* **Attention Mechanisms**: Scaled dot-product, causal (masked), and multi-head attention layers.
* **Model Architecture**: Layer normalization, GELU activation functions, feed-forward networks, and transformer blocks with shortcut connections.

### Key Features
* **Custom GPT Architecture**: Implements the GPT-2 124M parameter configuration with a 50,257-token vocabulary and 1,024-token context length.
* **Causal Masking**: Ensures the model can only attend to previous tokens during training, which is essential for generative tasks.
* **Residual Connections**: Uses shortcut connections to improve gradient flow in deep networks.
* **Modular Design**: Key components are abstracted into classes (e.g., `FeedForward`, `MultiHeadAttention`) for clarity and reuse.

### Setup and Requirements
This project requires Python 3.13 and the libraries listed in `requirements.txt`, including:
* **PyTorch**: Core deep learning framework.
* **tiktoken**: OpenAI's BPE tokenizer used for GPT-2/3.
* **Matplotlib**: Used for visualizing activation functions like GELU and ReLU.

To install dependencies:
```bash
pip install -r requirements.txt
```

### Acknowledgments
This project was developed as a personal learning initiative to master the internal workings of Large Language Models, aligning with my studies at the Polytechnic of Milan and my passion for Deep Learning. Special thanks to Sebastian Raschka for the foundational guide provided in Build a Large Language Model From Scratch.
