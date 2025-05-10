import os
import datasets
from tqdm import tqdm

os.makedirs("images", exist_ok=True)

dataset = datasets.load_dataset("macabdul9/LLaVA-CoT-o1-eCoT", split="train")

# import pdb;pdb.set_trace()
sft_data = []

for idx in tqdm(range(len(dataset))):
    example = dataset[idx]
    
    filename = "_".join(example["id"].split("/"))
    
    # Save the image to the local directory
    image_path = os.path.join("images", filename)
    example['image'].save(image_path)
    
    
    sft_data.append({
        "message": [
            {
                "content": example["question"] + "<image>",
                "role": "user",
            },
            {
                "content": example["ecot"],
                "role": "assistant",
            },
        ],
        "image": image_path,
    })
    if idx == 1000:
        print(f"Processed {idx} examples")
        break
    
    
# Save the data to a JSON file
import json
with open("sft_data.json", "w") as f:
    json.dump(sft_data, f, indent=4)