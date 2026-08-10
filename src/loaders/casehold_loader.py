import pandas as pd
import re


def load_casehold(file_path):
    """
    Load the CaseHOLD dataset.

    Returns a list of dictionaries.
    """

    df = pd.read_csv(file_path, engine="python")

    records = []

    for idx, row in df.iterrows():

        choices = re.findall(r"'([^']*)'", row["endings"])

        record = {
            "id": idx,
            "dataset": "CaseHOLD",
            "domain": "Legal",
            "context": row["context"],
            "choices": choices,
            "answer": choices[int(row["label"])],
            "label": int(row["label"])
        }

        records.append(record)

    return records