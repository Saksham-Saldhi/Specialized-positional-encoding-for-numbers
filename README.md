Project Description

Overview

This project explores a novel approach to positional embeddings for enhancing the numerical reasoning capabilities of language models. Specifically, the project focuses on modifying the positional embeddings for numerical tokens to better represent their place values, thus aiding the model in understanding and processing numerical data more effectively.

Motivation

Numerical reasoning is a critical aspect of many real-world applications, such as financial analysis, scientific computation, and data-driven decision-making. Traditional language models often struggle with numerical data, as they treat numbers similarly to other tokens, without leveraging the inherent structure and place value of digits. By customizing positional embeddings, the goal of this project is to provide language models with a clearer understanding of numerical structures, potentially improving their performance in tasks involving arithmetic, comparisons, and numerical word problems.

Baseline

The model that is suggested by Huiyin for your starting point is GPT-NeoX from Phytia. Please check its different models and see which of your models can you use using your available GPU resources.
Approach

The core idea involves assigning custom positional embeddings to digits in numbers, there are different ways to do this.

For the starting point we can explore a combination of reverse positional embedding for digits before the decimal point and negative values for digits after the decimal point

Reverse Positional Embedding for Integers:

For digits before the decimal point, positional embeddings are assigned in reverse order. For instance, in the number 123, the digit 1 is assigned a positional embedding of 3, the digit 2 is assigned 2, and the digit 3 is assigned 1.
Sequential Positional Embedding for Decimals:

For digits after the decimal point, positional embeddings are assigned sequentially with negative indices. For example, in the number 123.45, the digit 4 (tenths place) is assigned -1, and the digit 5 (hundredths place) is assigned -2.
Custom Tokenization:

The tokenizer automatically wraps numbers with special tokens [NUM] and [/NUM] to clearly delineate numerical sequences within the text. It then splits these sequences into individual digits and decimal points.
Implementation

The project involves modifying a pre-trained GPT-NeoX model to incorporate the custom positional embeddings. The key steps include:

Custom Tokenizer:

Extend the tokenizer to automatically identify and wrap numbers with [NUM] and [/NUM], and split them into individual components (digits and decimal points).
Custom Positional Embedding Layer:

Implement a custom positional embedding mechanism within the model to handle the new positional encoding scheme for numerical tokens.
Training and Fine-tuning:

Fine-tune the modified model on a dataset containing numerical data to enable the model to learn and leverage the new positional embeddings effectively.
Extended Pretraining

To adapt the model to work with the new positional embedding, we need to do an extended pretraining step, to train the model with the language modeling loss for a few steps.
@Huiyin Xue Could you please advise how much data and how many steps we should use for this?

Evaluation

Finally, We will finetune and evaluate the model on two datasets: (1) DROP, and (2) FERMAT.

