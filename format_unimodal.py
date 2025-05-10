import os
import json
import datasets
from tqdm import tqdm

# Load dataset
dataset = datasets.load_dataset("macabdul9/OpenMathReasoning-125K", split="train")

# Output lists
sft_data = []
dpo_data = []

# Helper to clean LaTeX escape sequences
def clean_latex(s):
    return s.replace("\\\\", "\\")

# Process entries
for idx in tqdm(range(len(dataset))):
    example = dataset[idx]

    # Clean all fields that may contain LaTeX
    problem = clean_latex(example["problem"])
    ecot = clean_latex(example["ecot"])
    long_cot = clean_latex(example["long_cot"])

    # SFT data format
    sft_data.append({
        "instruction": problem,
        "input": "",
        "output": ecot,
    })

    # DPO data format
    dpo_data.append({
        "conversations": [
            {
                "from": "human",
                "value": problem,
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": ecot,
        },
        "rejected": {
            "from": "gpt",
            "value": long_cot,
        }
    })

    if idx == 1000:
        print(f"Processed {idx} examples")
        break

# Save cleaned SFT and DPO data
os.makedirs("data", exist_ok=True)
with open("data/sft_data_unimodal_test.json", "w", encoding="utf-8") as f:
    json.dump(sft_data, f, indent=4, ensure_ascii=False)
with open("data/dpo_data_unimodal_test.json", "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, indent=2, ensure_ascii=False)

print("✅ Data saved successfully.")
