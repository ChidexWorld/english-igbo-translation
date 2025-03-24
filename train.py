import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Check if a checkpoint exists
checkpoint_dir = "./checkpoints"
last_checkpoint = None


# Load the MarianMT model for English-Igbo translation
model_name = "Helsinki-NLP/opus-mt-en-ig"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    last_checkpoint = max(
        [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)],
        key=os.path.getctime,  # Get the most recent checkpoint
    )

# Load the dataset
dataset = load_dataset("Tommy0201/igbo-to-english")

# Select only 2,000 samples from the dataset
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))

# Print an example
example = small_dataset[0]
print("Sample Example:", example)

# Check dataset splits
print(dataset)

# Split the dataset (90% train, 10% test)
split_dataset = small_dataset.train_test_split(test_size=0.1, seed=42)


def preprocess_function(examples):
    inputs = tokenizer(
        examples["english"], padding="max_length", truncation=True, max_length=128
    )
    targets = tokenizer(
        examples["igbo"], padding="max_length", truncation=True, max_length=128
    )
    inputs["labels"] = targets["input_ids"]

    return inputs


# Apply preprocessing
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

# Define training settings
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    fp16=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("test"),
)

# Train (Resume from checkpoint if available)
if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(last_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch.")
    trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Reload fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")


# Test the Model
def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# dynamically using translation
while True:
    english_text = input("Enter English text to translate (or type 'exit' to quit): ")
    if english_text.lower() == "exit":
        print("Exiting...")
        break

    igbo_translation = translate(english_text)
    print("English:", english_text)
    print("Igbo:", igbo_translation)
    print("-" * 50)  # Separator for readability
