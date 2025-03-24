English to Igbo Translation Model Training and Usage

A machine translation model for translating English to Igbo, based on the Helsinki-NLP/opus-mt-en-ig model.

Overview

This project provides a full pipeline for training an English-to-Igbo translation model. It covers:
✅ Dataset loading and preprocessing
✅ Model training and checkpointing
✅ Saving and reloading fine-tuned models
✅ Dynamic text translation

Installation

Ensure you have Python 3.8+ and install the required libraries:

pip install transformers datasets torch 

1. Import Required Libraries

from transformers import ( AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, ) from datasets import load_dataset import os 

Key components:

AutoTokenizer – Loads the tokenizer.

AutoModelForSeq2SeqLM – Loads the translation model.

TrainingArguments – Defines training configurations.

Trainer – Manages model training.

load_dataset – Fetches the dataset.

2. Check for Existing Checkpoints

checkpoint_dir = "./checkpoints" last_checkpoint = None if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir): last_checkpoint = max( [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)], key=os.path.getctime, # Get the most recent checkpoint ) 

Purpose:
✔️ Resumes training from the latest checkpoint if available.

3. Load the Pretrained Model and Tokenizer

model_name = "Helsinki-NLP/opus-mt-en-ig" tokenizer = AutoTokenizer.from_pretrained(model_name) model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 

✔️ Loads the MarianMT model for English-to-Igbo translation.

4. Load the Dataset

dataset = load_dataset("Tommy0201/igbo-to-english") example = dataset["train"][0] print(example) print(dataset) 

✔️ Fetches an English-Igbo translation dataset.

5. Split the Dataset

split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42) 

✔️ Splits the dataset into 90% training and 10% testing.

6. Preprocess the Data

def preprocess_function(examples): inputs = tokenizer( examples["english"], padding="max_length", truncation=True, max_length=128 ) targets = tokenizer( examples["igbo"], padding="max_length", truncation=True, max_length=128 ) inputs["labels"] = targets["input_ids"] return inputs tokenized_datasets = split_dataset.map(preprocess_function, batched=True) 

✔️ Tokenizes and prepares input-output pairs for training.

7. Define Training Arguments

training_args = TrainingArguments( output_dir="./checkpoints", eval_strategy="epoch", save_strategy="epoch", per_device_train_batch_size=8, per_device_eval_batch_size=8, num_train_epochs=3, logging_dir="./logs", logging_steps=500, save_total_limit=2, ) 

✔️ Defines batch size, evaluation strategy, and logging settings.

8. Initialize Trainer

trainer = Trainer( model=model, args=training_args, train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets.get("test"), ) 

✔️ Configures the Trainer for managing model training.

9. Train the Model

if last_checkpoint: print(f"Resuming training from checkpoint: {last_checkpoint}") trainer.train(last_checkpoint) else: print("No checkpoint found. Starting training from scratch.") trainer.train() 

✔️ Starts training or resumes from a checkpoint.

10. Save and Reload the Fine-Tuned Model

model.save_pretrained("./fine_tuned_model") tokenizer.save_pretrained("./fine_tuned_model") model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model") tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model") 

✔️ Saves and reloads the trained model for translation tasks.

11. Translate English to Igbo

def translate(text): inputs = tokenizer(text, return_tensors="pt") output = model.generate(**inputs) return tokenizer.decode(output[0], skip_special_tokens=True) 

✔️ Defines a function to translate English text to Igbo.

12. Dynamic User Input for Translation

while True: english_text = input("Enter English text to translate (or type 'exit' to quit): ") if english_text.lower() == "exit": print("Exiting...") break igbo_translation = translate(english_text) print("English:", english_text) print("Igbo:", igbo_translation) print("-" * 50) # Separator for readability 

✔️ Allows users to enter text dynamically for translation.

Usage Example

Run the script and enter English text for translation:

python translate.py 

Example Input:

Hello, how are you? 

Output:

Ndewo, kedu ka i mere? 

Conclusion

This project provides a complete pipeline for training and using an English-to-Igbo translation model using Hugging Face Transformers. 🚀

✔️ Pretrained model for quick translation
✔️ Fine-tuning support for better accuracy
✔️ Interactive CLI input for real-time translation

License

📜 This project is open-source under the MIT License.


