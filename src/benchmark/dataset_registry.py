from loaders.cuad_loader import load_cuad
from loaders.casehold_loader import load_casehold
from loaders.medqa_loader import load_medqa
from loaders.pubmedqa_loader import load_pubmedqa
from loaders.qasper_loader import load_qasper
from loaders.sciq_loader import load_sciq


DATASETS = [

    {
        "name": "CUAD",
        "domain": "Legal",
        "loader": load_cuad,
        "path": "../datasets/Legal/CUAD/CUADv1.json"
    },

    {
        "name": "CaseHOLD",
        "domain": "Legal",
        "loader": load_casehold,
        "path": "../datasets/Legal/LexGLUE/case_hold/case_hold_train.csv"
    },

    {
        "name": "MedQA",
        "domain": "Medical",
        "loader": load_medqa,
        "path": "../datasets/Medical/MedQA/questions/dev.jsonl"
    },

    {
        "name": "PubMedQA",
        "domain": "Medical",
        "loader": load_pubmedqa,
        "path": "../datasets/Medical/PubMedQA/pqa_labeled/train_labeled.csv"
    },

    {
        "name": "QASPER",
        "domain": "Scientific",
        "loader": load_qasper,
        "path": "../datasets/Scientific/qasper/train.parquet"
    },

    {
        "name": "SciQ",
        "domain": "Scientific",
        "loader": load_sciq,
        "path": "../datasets/Scientific/SciQ/train.csv"
    }

]