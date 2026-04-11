import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, precision_recall_curve, average_precision_score
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# --- Configuration ---
MODEL_DIR = "./claim_detection_model"
DATA_FILE = "./datasets/2xNCS.json"

def load_test_data(json_path):
    """Recreates the exact 20% test split used during training."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)[['text', 'label']]
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return test_df

def main():
    print("Loading test data...")
    test_df = load_test_data(DATA_FILE)

    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing test data...")
    test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=16,
        use_cpu=True,
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args)

    print("Running predictions...")
    prediction_output = trainer.predict(test_ds)
    

    logits = prediction_output.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    labels = prediction_output.label_ids

    # --- Metrics Calculation ---
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    print("\n" + "="*30)
    print("📊 MODEL EVALUATION METRICS 📊")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("="*30 + "\n")


    print("Generating visualizations...")

    # 1. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Claim (0)', 'Claim (1)'], 
                yticklabels=['Not Claim (0)', 'Claim (1)'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Claim Detection')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved -> confusion_matrix.png")

    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2, 
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300)
    plt.close()
    print("Saved -> precision_recall_curve.png")

if __name__ == "__main__":
    main()