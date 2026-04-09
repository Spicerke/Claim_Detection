import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)


# Mapping: 0: False (NFS), 1: True & Worthy (CFS), 2: True but Unworthy (UFS)
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./claim_detection_model"

# Note: In a real run, you would download the 'claim_labels.csv' from the 
# official ClaimBuster repo or Zenodo.
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    label_map = {"NFS": 0, "CFS": 1, "UFS": 2}
    df['label'] = df['verdict'].map(label_map)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Metrics Calculation (F1-Score)
metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Using 'macro' F1 to treat all classes equally regardless of frequency
    return metric.compute(predictions=predictions, references=labels, average="macro")

# 4. Training Execution
def train_model(train_df, test_df):
    # Convert pandas to Hugging Face Dataset objects
    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    # Load DistilBERT for 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_cpu=True # Force CPU 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # Save the model weights for the API
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Ensure you have the dataset file ready
    # train_df, test_df = load_and_preprocess_data("claimbuster_data.csv")
    # train_model(train_df, test_df)
    pass