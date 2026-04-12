import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./claim_detection_model"
DATA_FILE = "./datasets/balanced_training_data.json"

def load_and_preprocess_data(json_path):
    """Loads the ClaimBuster JSON and splits it 80/20."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    df = df[['text', 'label']]
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    return train_df, test_df

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    # Truncate and pad to a max length of 128 tokens to save memory and speed up CPU training
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(eval_pred):
    """Calculates F1, Accuracy, Precision, and Recall for the evaluation phase."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = f1_score(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    return {
        "accuracy": acc, 
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    # Load and split data
    train_df, test_df = load_and_preprocess_data(DATA_FILE)

    # Convert Pandas DataFrames to Hugging Face Dataset objects
    print("Tokenizing datasets...")
    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    # Load DistilBERT for Binary Classification (num_labels=2)
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  
        save_strategy="epoch",        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,           
        weight_decay=0.01,
        load_best_model_at_end=True,  
        use_cpu=True,                 
        report_to="none"            
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )


    print("Starting training loop...")
    trainer.train()
    
    # Save the final, best model 
    print(f"Saving finalized model and tokenizer to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete! Your model is ready for the API.")

if __name__ == "__main__":
    main()