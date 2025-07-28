#!/usr/bin/env python
"""
Fine-tuning of meta-llama/Llama-4-Maverick-17B-128E-Original for text classification.

This script demonstrates how to load the Llama-4 Maverick model, preprocess the dataset, 
and fine-tune it for a text classification task.

Requirements:
    - Python 3.8+
    - transformers (see requirements.txt)
    - torch (see requirements.txt)
    - datasets (see requirements.txt)
    
Note:
    This model is large (17B parameters), ensure you have adequate hardware (high-memory GPU).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

# Set your Hugging Face authentication token here:
auth_token = "your_token_here"  # Replace with your Hugging Face token or use CLI authentication

# Load the tokenizer and model for Llama-4 Maverick
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Original", use_auth_token=auth_token)
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Original", num_labels=2, use_auth_token=auth_token)

def preprocess_function(examples):
    """
    Tokenizes and processes the dataset to fit the model's input format.
    """
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def main():
    # Load a sample dataset (you can replace this with your own dataset)
    dataset = load_dataset("imdb", split="train[:10%]")
    
    # Preprocess the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Create a DataLoader for batch processing
    train_dataloader = DataLoader(tokenized_datasets, batch_size=8)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir='./results',               # Output directory to save model checkpoints
        num_train_epochs=3,                   # Number of training epochs
        per_device_train_batch_size=2,        # Batch size per device (adjust based on GPU memory)
        warmup_steps=500,                     # Number of warmup steps
        weight_decay=0.01,                    # Strength of weight decay
        logging_dir='./logs',                 # Directory to save logs
        logging_steps=200,                    # Log every 200 steps
        save_steps=1000,                      # Save model checkpoint every 1000 steps
        evaluation_strategy="epoch",          # Evaluate every epoch
        save_total_limit=3,                   # Limit the number of saved checkpoints
        load_best_model_at_end=True          # Load the best model based on evaluation
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,  # For evaluation, we are using the same dataset
        data_collator=None,               # You can use DataCollatorForPadding for dynamic padding
        tokenizer=tokenizer,
    )

    # Start the fine-tuning process
    print("Starting fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_llama4_maverick_17B_model')

    # Test the fine-tuned model with a sample input
    test_text = "The movie was fantastic with great performances and stunning visuals."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)

    print(f"Prediction: {prediction.item()}")

if __name__ == "__main__":
    main()
