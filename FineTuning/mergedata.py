import pandas as pd
import json

# --- Configuration ---
ORIGINAL_DATA = "./datasets/3xNCS.json"
FEVER_DATA = "./datasets/fever_claims_cleaned.json"
OUTPUT_DATA = "./datasets/balanced_training_data.json"

def main():
    print(f"Loading original dataset: {ORIGINAL_DATA}")
    df_original = pd.read_json(ORIGINAL_DATA)
    
    # Ensure labels are integers
    df_original['label'] = df_original['label'].astype(int)

    # 1. Calculate current class distribution
    counts = df_original['label'].value_counts()
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)

    print("\nCurrent Original Dataset Distribution:")
    print(f"Not a Claim (0): {count_0}")
    print(f"Factual Claim (1): {count_1}")

    # 2. Determine how many FEVER claims to add
    print(f"\nLoading cleaned FEVER data: {FEVER_DATA}")
    df_fever = pd.read_json(FEVER_DATA)
    
    if count_0 > count_1:
        # We have more 0s than 1s, so we add enough 1s to make them exactly equal
        needed_claims = count_0 - count_1
        print(f"\nTargeting a 50/50 balance. We need {needed_claims} more claims (1s).")
        
        # Guardrail in case FEVER is smaller than what we need
        if needed_claims > len(df_fever):
            print(f" We only have {len(df_fever)} FEVER claims available! Adding all of them.")
            sampled_fever = df_fever
        else:
            print(f"Randomly sampling {needed_claims} claims from the FEVER dataset...")
            sampled_fever = df_fever.sample(n=needed_claims, random_state=42)
            
        # 3. Merge datasets
        df_combined = pd.concat([df_original, sampled_fever], ignore_index=True)
        
    else:
        print("\nYour dataset already has more claims than non-claims!")
        # Fallback: Just add 20% more to introduce domain diversity without breaking balance entirely
        add_amount = int(count_1 * 0.20)
        print(f"Adding {add_amount} FEVER claims just to increase general-knowledge diversity.")
        sampled_fever = df_fever.sample(n=add_amount, random_state=42)
        df_combined = pd.concat([df_original, sampled_fever], ignore_index=True)

    # 4. Shuffle the combined dataset
    # This ensures the model doesn't learn all ClaimBuster data first, and all FEVER data second
    print("Shuffling the merged dataset...")
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Save the final dataset
    # We only keep 'text' and 'label' columns to match your training script exactly
    print(f"\nSaving final balanced dataset to {OUTPUT_DATA}...")
    df_combined[['text', 'label']].to_json(OUTPUT_DATA, orient='records', indent=4)
    
    # Final sanity check printout
    final_counts = df_combined['label'].value_counts()
    print("\n✅ Merge Complete! Final Dataset Distribution:")
    print(f"Not a Claim (0): {final_counts.get(0, 0)}")
    print(f"Factual Claim (1): {final_counts.get(1, 0)}")
    print(f"Total Rows:      {len(df_combined)}")

if __name__ == "__main__":
    main()