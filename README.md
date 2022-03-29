# tokenizers_datasets_transformers
Fast State-of-the-art tokenizers, optimized for both research and production;



##### Intro_Tokenizers.ipynb

this file is to use a corpus text, create tokenizer architecture, train Tokenizer on corpus and get specific Tokenizer for your text corpus.

------

[Quicktour](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)

To build a tokenizer from scratch for a specific subject.

- Start with all the characters present in the training corpus as tokens.
- Identify the most common pair of tokens and merge it into one token.
- Repeat until the vocabulary (e.g., the number of tokens , 30k, 60k e 90k) has reached the size we want. 



***class*tokenizers.trainers.BpeTrainer[](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.trainers.BpeTrainer)**

Trainer capable of training a BPE model

- Parameters

  **vocab_size** (`int`, optional) – The size of the final vocabulary, including all tokens and alphabet.

  **min_frequency** (`int`, optional) – The minimum frequency a pair should have in order to be merged.

  **show_progress** (`bool`, optional) – Whether to show progress bars while training.

  **special_tokens** (`List[Union[str, AddedToken]]`, optional) – A list of special tokens the model should know of.

  **limit_alphabet** (`int`, optional) – The maximum different characters to keep in the alphabet.

  **initial_alphabet** (`List[str]`, optional) – A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept.

  **continuing_subword_prefix** (`str`, optional) – A prefix to be used for every subword that is not a beginning-of-word.

  **end_of_word_suffix** (`str`, optional) – A suffix to be used for every subword that is a end-of-word

  

<h5>Intro_Datasets.ipynb</h5>
------

https://huggingface.co/docs/datasets/v1.13.0/quickstart.html

The quick start is intended for developers who are ready to dive in to the code, and see an end-to-end example of how they can integrate 🤗 Datasets into their model training workflow. 

https://huggingface.co/docs/datasets/tutorial

# Overview

#### Welcome to the 🤗 Datasets tutorial!

The goal of the tutorials are to help new users build up a basic understanding of 🤗 Datasets. You will learn to:

- Setup a virtual environment and install 🤗 Datasets.
- Load a dataset.
- Explore what’s inside a Dataset object.
- Use a dataset with PyTorch and TensorFlow.
- Evaluate your model predictions with a metric.
- Easily upload a dataset to the Hugging Face Hub.



## **intro_transformers.ipynb**

State-of-the-art Machine Learning for PyTorch, TensorFlow and JAX.



🤗 Transformers provides APIs to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you time from training a model from scratch. The models can be used across different modalities such as:

- 📝 Text: text classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages.
- 🖼️ Images: image classification, object detection, and segmentation.
- 🗣️ Audio: speech recognition and audio classification.
- 🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.
