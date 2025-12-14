# Toward Reliable Scientific Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://arxiv.org/abs/2505.14599)
[![Dataset](https://img.shields.io/badge/dataset-available-yellow)](https://huggingface.co/TruthHypo)

## News
- Our paper is accepted to IJCAI 2025!

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Structure](#structure)
- [Citation](#citation)

## Introduction
TruthHypo is a benchmark for assessing the capabilities of LLMs in generating truthful scientific hypotheses. This repo also contains the source code of KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our [paper](https://arxiv.org/abs/2505.14599) shows that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs.

## Usage
The TruthHypo dataset is directly accessible via [HuggingFace](https://huggingface.co/TruthHypo):
```python
from datasets import load_dataset

data = load_dataset("TruthHypo/edges_test")
```

The processed knowledge sources for knowledge-enhanced hypothesis generation can be found at
- Literature
  - [PubMed Articles](https://huggingface.co/datasets/MedRAG/pubmed)
- Knowledge Graph
  - [PubTator Edges](https://huggingface.co/datasets/TruthHypo/edges_train)
  - [PubTator Nodes](https://huggingface.co/datasets/TruthHypo/nodes)

## Structure

Our repository contains the following contents:
- data: the data of TruthHypo benchmark
  - edges_test.tsv: the test data used for LLM evaluation
- src: the source code of agents and verifiers used in our experiments
  - agent: the LLM agents used to generated biomedical hypotheses
    - base.py: the base agent
    - cot.py: the agent using parametric knowledge only
    - kg.py: the agent using both parametric knowledge and information fromknowledge graphs
    - rag.py: the agent using both parametric knowledge and information from scientific literature
    - rag_kg.py: the agent using parametric knowledge and information from both knowledge graphs and scientific literature
  - verifier: the LLM verifiers used to measure the groundedness of generated hypotheses
    - rag_verifier.py: the verifier with scientific literature as the supporting knowledge base
    - kg_verifier.py: the verifier with knowledge graphs as the supporting knowledge base
    - rag_kg_verifier.py: the verifier with both scientific literature and knowledge graphs as the supporting knowledge base


## Citation
```
@inproceedings{xiong2025toward,
  author       = {Guangzhi Xiong and
                  Eric Xie and
                  Corey Williams and
                  Myles Kim and
                  Amir Hassan Shariatmadari and
                  Sikun Guo and
                  Stefan Bekiranov and
                  Aidong Zhang},
  title        = {Toward Reliable Scientific Hypothesis Generation: Evaluating Truthfulness
                  and Hallucination in Large Language Models},
  booktitle    = {Proceedings of the Thirty-Fourth International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2025, Montreal, Canada, August 16-22,
                  2025},
  pages        = {7849--7857},
  publisher    = {ijcai.org},
  year         = {2025},
  url          = {https://doi.org/10.24963/ijcai.2025/873},
  doi          = {10.24963/IJCAI.2025/873},
  timestamp    = {Wed, 24 Sep 2025 17:45:28 +0200},
  biburl       = {https://dblp.org/rec/conf/ijcai/XiongXWKSGBZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
