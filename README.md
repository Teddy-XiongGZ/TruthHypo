# Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models

Here is the repository of our paper: *Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models*.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://arxiv.org/abs/2505.14599)

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
@article{xiong2025toward,
      title={Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models}, 
      author={Guangzhi Xiong and Eric Xie and Corey Williams and Myles Kim and Amir Hassan Shariatmadari and Sikun Guo and Stefan Bekiranov and Aidong Zhang},
      journal={arXiv preprint arXiv:2505.14599},
      year={2025}
}
```
