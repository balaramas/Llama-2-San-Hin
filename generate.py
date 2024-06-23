import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "balaramas/llama-2-7b-sahikpsir2"

# LoRA attention dimension
lora_r = 64  # 64 is rank, kind of hyperparameter

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load dataset (you can process it here)
#dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


import csv

# Initialize lists to store the elements of the first and second columns
first_column_elements = []
second_column_elements = []

# Assuming your CSV file is named 'data.csv' and it's in the current directory
file_path = './csv_files/test_sahi.csv'

# Read the CSV file and extract the elements of the first and second columns
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        first_column_elements.append(row[0])
        second_column_elements.append(row[1])


from tqdm import tqdm
generated_hin_preds=[]

for i in tqdm(range(len(first_column_elements)),desc="Running on test split"):
  # Run text generation pipeline with our next model
  prompt = f"translate this text {first_column_elements[i]} from Sanskrit to Hindi"
  pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=50)
  result = pipe(f"<s>[INST] {prompt} [/INST]")
  #print(result[0]['generated_text'])
  given_string=result[0]['generated_text']
  start_index = given_string.find("[/INST]")
  extracted_part = given_string[(8+start_index):]
  modified_string = extracted_part.replace("।", "")
  generated_text = modified_string.strip()
  generated_text = generated_text+" ।"
  generated_hin_preds.append(generated_text)


import csv

# Path to the CSV file
file_path = './csv_files/generated_preds3.csv'

# Write the elements of the list to the CSV file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for element in generated_hin_preds:
        writer.writerow([element])

print("Elements have been written to", file_path)


