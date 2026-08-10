import json


def load_cuad(file_path):
    """
    Load the CUAD dataset.

    Returns a list of dictionaries.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for contract in data["data"]:

        title = contract["title"]

        for paragraph in contract["paragraphs"]:

            context = paragraph["context"]

            for qa in paragraph["qas"]:

                answers = [a["text"] for a in qa["answers"]]

                record = {
                    "id": qa["id"],
                    "dataset": "CUAD",
                    "domain": "Legal",
                    "title": title,
                    "question": qa["question"],
                    "context": context,
                    "answers": answers,
                    "is_impossible": qa["is_impossible"]
                }

                records.append(record)

    return records