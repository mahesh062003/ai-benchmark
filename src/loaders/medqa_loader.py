import json


def load_medqa(file_path):
    """
    Load the MedQA dataset.

    Returns a list of dictionaries.
    """

    records = []

    with open(file_path, "r", encoding="utf-8") as f:

        for idx, line in enumerate(f):

            row = json.loads(line)

            record = {
                "id": idx,
                "dataset": "MedQA",
                "domain": "Medical",
                "question": row["question"],
                "options": row["options"],
                "answer": row["answer"],
                "answer_idx": row["answer_idx"],
                "meta_info": row.get("meta_info", "")
            }

            records.append(record)

    return records