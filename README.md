# LLAMA3_for_coding
Fine tuning LLAMA3 on coding instructional dataset

Hugingface dataset: https://huggingface.co/datasets/dinhlnd1610/CodeAlpaca-AddLanguage

- I added programming language for CodeApaca Dataset: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k and filtered some un-unused languages.
- To add programming language, I used column output of CodeAlpaca Dataset and used model : https://huggingface.co/philomath-1209/programming-language-identification to compute logits. To make the inference stage faster, I used multiprocessing library of Pytorch
- After processing, there are around 10k samples
