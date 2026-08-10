import pandas as pd


def load_sciq(file_path):
    """
    Loads the SciQ dataset.

    Returns:
        List[Dict]
    """

    df = pd.read_csv(file_path)

    dataset = []

    for idx, row in df.iterrows():

        options = {
            "A": row["correct_answer"],
            "B": row["distractor1"],
            "C": row["distractor2"],
            "D": row["distractor3"],
        }

        dataset.append({

            "id": idx,

            "dataset": "SciQ",

            "domain": "Scientific",

            "question": row["question"],

            "context": row["support"],

            "options": options,

            "answer": row["correct_answer"],

            "answer_idx": "A"

        })

    return dataset