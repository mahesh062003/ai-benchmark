import json
import os


def save_chunks(chunks, dataset_name):

    output_dir = "../results/chunks"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{dataset_name}_chunks.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            chunks,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved {len(chunks)} chunks")
    print(f"Location: {output_path}")