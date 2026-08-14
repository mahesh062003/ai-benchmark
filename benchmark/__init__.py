"""Corpus construction, index building and the retrieval benchmark runner.

The pipeline is corpus -> indexes -> benchmark. Each stage persists its output
to the artifacts directory so a later stage can resume without repeating work.
"""
