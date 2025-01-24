import os
import re
import torch
import openai
import transformers
from liquid import Template
from src.agent.base import BaseAgent
from src.IR import RetrievalSystem
from src.KGR import KnowledgeGraphRetriever

class KGVerifier(BaseAgent):
    def __init__(self, model_name="OpenAI/gpt-4o-mini", cache_dir="../huggingface/hub", model_dtype=torch.bfloat16, api_key=None, pmid_cutoff=36600000, retriever_name="BM25", corpus_name="PubMed", HNSW=True, model=None, retrieval_system=None, kg_retriever=None, **kwargs):
        super().__init__(model_name, cache_dir, model_dtype, api_key, model)

        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.hnsw = HNSW
        self.retrieval_system = retrieval_system or None
        self.pmid_cutoff = pmid_cutoff
        self.kg_retriever = kg_retriever or KnowledgeGraphRetriever(**kwargs)
        
    def verify_claim(self, claim, temperature=0.0, max_new_tokens=2048, k=8, max_n_edges=100, node_k=100, seed=0, nodes=None, edges=None, documents=None, **kwargs):
        # if documents is None:
        #     if self.retrieval_system is None:
        #         self.retrieval_system = RetrievalSystem(retriever_name=self.retriever_name, corpus_name=self.corpus_name, cache=True, HNSW=self.hnsw)
        #     documents = self.retrieval_system.retrieve(claim, k=max(32, 2 * k))[0]
        #     documents = [doc for doc in documents if doc["PMID"] < self.pmid_cutoff][:k]
        # document_text = '\n'.join(["[Title: {:s}] {:s}".format(doc["title"], doc["content"]) for doc in documents])

        if edges is None:
            if nodes is None:
                try:
                    entities = eval(
                        re.findall(
                            r'```json\s*({(?:[^`]|\`(?!``))*})', 
                            self.call_llm(
                                [
                                    {
                                        "role": "user",
                                        "content": f"### Claim\n{claim}\n\nExtract key entities from the claim that will be used to search for relevant information in an external knowledge graph. Each entity should be extracted as \"entity_type (e.g., Disease/Chemical/Gene/Mutation) entity_name (entity_id if presented)\". Output the extracted entities in the JSON format: ```json{{\"entities\": [\"entity1\", ...]}}```"
                                    }
                                ]
                            )[0], 
                            re.DOTALL
                        )[-1]
                    )["entities"]
                except:
                    entities = [claim]
                nodes = [self.kg_retriever.retrieve_nodes(text, k=node_k) for text in entities]
                nodes = [n for ns in nodes for n in ns]
            node_ids = [f"{n['Type']}|{n['ConceptID']}" for n in nodes]
            edges = self.kg_retriever.edges[self.kg_retriever.edges["x_id"].isin(node_ids) * self.kg_retriever.edges["y_id"].isin(node_ids)].to_dict('records')

            if len(edges) > max_n_edges:
                import random; random.seed(seed)
                edges = random.sample(edges, max_n_edges)

        edge_text = '\n'.join([self.kg_retriever.edge2text(e) for e in edges])
        
        messages = [
            {
                "role": "system",
                "content": "You are a scientist. Your task is to verify if the relevant knowledge (if applicable) can support the given claim."
            },
            {
                "role": "user",
                # "content": f"### Relevant Documents\n{document_text}\n\n### Relevant Knowledge\n{edge_text}### Claim\n{claim}\n\nOutput {{\"groundness\": 1}} if the materials support the claim. Otherwise, output {{\"groundness\": 0}}."
                "content": f"### Relevant Knowledge\n{edge_text}\n\n### Claim\n{claim}\n\nJudge if the given information supports the claim. Output {{\"groundness\": 1}} if the materials support the claim else {{\"groundness\": 0}}."
            }
        ]
        groundness = 1 if "1" in self.call_llm(messages, temperature=temperature, num_return_sequences=1, max_new_tokens=max_new_tokens)[0] else 0
        return groundness, documents, nodes, edges
    
    def verify_claims(self, claims: str | list[str], temperature=0.0, max_new_tokens=2048, k=8, max_n_edges=100, node_k=100, seed=0, max_n_claims=10, nodes=None, edges=None, documents=None, **kwargs):
        if type(claims) == str:
            try:
                claims = eval(
                    re.findall(
                        r'```json\s*({(?:[^`]|\`(?!``))*})', 
                        self.call_llm(
                            [
                                {
                                    "role": "user",
                                    "content": f"### Statement\n{claims}\n\n Summarize the statement as a list of claims which will be further verified by external resources. Output the summarized claims in the JSON format: ```json{{\"claims\": [\"claim1\", ...]}}```"
                                }
                            ]
                        )[0], 
                        re.DOTALL
                    )[-1]
                )["claims"]
            except:
                claims = re.sub(r'\n\n', '\n', claims.strip(':').strip()).split('\n')
        # claims = claims[:max_n_claims]
        claims = claims[-max_n_claims:]
        nodes = nodes or dict()
        edges = edges or dict()
        documents = documents or dict()
        groundness = []
        for claim in claims:
            c_ground, c_docs, c_nodes, c_edges = self.verify_claim(claim, temperature, max_new_tokens, k, max_n_edges, node_k, seed, nodes=nodes.get(claim, None), edges=edges.get(claim, None), documents=documents.get(claim, None))
            groundness.append(c_ground)
            documents[claim] = c_docs
            nodes[claim] = c_nodes
            edges[claim] = c_edges
        return claims, groundness, documents, nodes, edges