import json

# Change these paths to where your FEVER data actually is
INPUT_FILE = "./datasets/train.jsonl"
OUTPUT_FILE = "./datasets/fever_claims_cleaned.json"

def process_fever_data():
    print(f"Reading FEVER data from {INPUT_FILE}...")
    cleaned_data = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            cleaned_data.append({
                "sentence_id": item["id"],
                "label": 1,
                "text": item["claim"],
            })
            
    print(f"Processed {len(cleaned_data)} claims.")
    
    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Save as a standard JSON array so your Pandas DataFrame can load it easily
        json.dump(cleaned_data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    process_fever_data()