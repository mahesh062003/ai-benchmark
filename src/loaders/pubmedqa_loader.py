import pandas as pd
import re


def load_pubmedqa(file_path):
    """
    Load the labelled PubMedQA dataset.

    Returns:
        list[dict]
    """

    df = pd.read_csv(file_path)

    records = []

    for _, row in df.iterrows():

        context_string = row["context"]

        # Extract the text inside the "contexts" list
        match = re.search(
            r"'contexts':\s*\[(.*?)\]\s*,\s*'labels'",
            context_string,
            flags=re.DOTALL
        )

        if match:
            contexts = re.findall(r"'([^']*)'", match.group(1))
        else:
            contexts = []

        record = {
            "id": row["pubid"],
            "dataset": "PubMedQA",
            "domain": "Medical",
            "question": row["question"],
            "context": contexts,
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"]
        }

        records.append(record)

    return records