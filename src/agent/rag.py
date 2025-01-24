import os
import torch
import openai
import transformers
from liquid import Template
from src.agent.base import BaseAgent
from src.IR import RetrievalSystem

class RAGAgent(BaseAgent):
    def __init__(self, model_name="OpenAI/gpt-4o-mini", cache_dir="../huggingface/hub", model_dtype=torch.bfloat16, api_key=None, pmid_cutoff=36600000, retriever_name="BM25", corpus_name="PubMed", HNSW=True, **kwargs):
        super().__init__(model_name, cache_dir, model_dtype, api_key)

        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.hnsw = HNSW
        self.retrieval_system = None
        self.pmid_cutoff = pmid_cutoff
        self.system_prompt = "You are a scientist. Your task is to generate a scientific hypothesis following given instructions."
        self.user_template = Template('''### Relevant Documents
{{document}}

### User Input
{{background}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
- Think step-by-step to derive the hypothesis.

2. **### Structured Output**:
- Present your proposed hypothesis in the following JSON format:
  ```json
  {
      "proposed_hypothesis": "Statement of the proposed hypothesis"
  }
  ```''')

    def generate_hypothesis(self, background, temperature=0.0, n_hypotheses=1, max_new_tokens=2048, k=32, documents=None, **kwargs):
        if documents is None:
            if self.retrieval_system is None:
                self.retrieval_system = RetrievalSystem(retriever_name=self.retriever_name, corpus_name=self.corpus_name, cache=True, HNSW=self.hnsw)
            documents = self.retrieval_system.retrieve(background, k=max(32, 2 * k))[0]
            documents = [doc for doc in documents if doc["PMID"] < self.pmid_cutoff][:k]
        document_text = '\n'.join(["[Title: {:s}] {:s}".format(doc["title"], doc["content"]) for doc in documents])
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_template.render(background=background, document=document_text)
            }
        ]
        return self.call_llm(messages, temperature=temperature, num_return_sequences=n_hypotheses, max_new_tokens=max_new_tokens), documents