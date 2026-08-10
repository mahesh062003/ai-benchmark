import sqlite3
import os


class BenchmarkDatabase:

    def __init__(self, db_path="../results/benchmark.db"):

        os.makedirs("../results", exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.create_table()

    def create_table(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            dataset TEXT,
            domain TEXT,
            retrieval_method TEXT,

            question TEXT,
            generated_answer TEXT,

            recall REAL,
            mrr REAL,
            ndcg REAL,

            faithfulness REAL,
            hallucination REAL,

            response_time REAL

        )
        """)

        self.conn.commit()

    def insert_result(

        self,

        dataset,
        domain,
        retrieval_method,
        question,
        generated_answer,
        recall,
        mrr,
        ndcg,
        faithfulness,
        hallucination,
        response_time

    ):

        self.cursor.execute("""

        INSERT INTO experiments(

            dataset,
            domain,
            retrieval_method,
            question,
            generated_answer,
            recall,
            mrr,
            ndcg,
            faithfulness,
            hallucination,
            response_time

        )

        VALUES(?,?,?,?,?,?,?,?,?,?,?)

        """, (

            dataset,
            domain,
            retrieval_method,
            question,
            generated_answer,
            recall,
            mrr,
            ndcg,
            faithfulness,
            hallucination,
            response_time

        ))

        self.conn.commit()

    def close(self):

        self.conn.close()