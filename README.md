# Llama-2-San-Hin

## Train LLM to perform machine translation (this repository contains code to format dataset to support training of Llama-2 )

## Install the requirements.txt files for dependencies and libraries (There can be other dependencies required based on your system).

## Data Preprocess 

```bash
python create_llm_dataset.py 
```

- Run create_llm_dataset.py to read dataset of MT from csv files in format given in directory "./csv_files/train_sahi.csv" and format to the guanaco dataset format that contains instructions and answers.
- The dataset is pushed to huggingface, edit the name of the saving file.
- In this case I saved the dataset in huggingface by the name balaramas/sahikpsir-train (private).

## Run the 'train_llm.py' file by specifying the name of the base llm model (in this case it "NousResearch/Llama-2-7b-chat-hf") and the name of the dataset (in this case it was "balaramas/sahikpsir-train" which is the directory name of the dataset saved in huggingface).

```bash
python train_llm.py 
```

- It will train the llm with QLoRa.
- Also it will push the model after training to huggingface, so login to huggingface and specify the access token of your account (check it up on google for logging huggingface cli).

## After the model is trained, it can generate result using the test split by runing the 'generate.py' where instead of train, specify the test split.

- The result saved then can be used to find the BLEU score using the sacrebleu from huggingface.

## .... BLEU score .... (Will be updated)
