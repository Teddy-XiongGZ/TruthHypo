import os
import json
import tqdm
import random
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

class KnowledgeGraphRetriever:
    def __init__(self, load_dir="./LinkHypoGen"):
        self.load_dir = load_dir
        self.nodes = pd.read_csv(os.path.join(self.load_dir, "nodes.tsv"), sep='\t')
        self.index_dir = os.path.join(self.load_dir, "node_index")
        if not os.path.exists(self.index_dir):
            mention_dir = os.path.join(self.load_dir, "node_mentions")
            if not os.path.exists(self.index_dir):
                os.makedirs(mention_dir, exist_ok=True)
                print("Saving entity mentions...")
                with open(os.path.join(mention_dir, "mentions.jsonl"), 'w') as f:
                    for i in tqdm.tqdm(range(len(self.nodes))):
                        # f.write(json.dumps({"id": i, "contents": " | ".join([j for m, c in eval(self.nodes.iloc[i]["mention_counts"]).items() for j in [m] * c])}) + '\n')
                        # f.write(json.dumps({"id": i, "contents": f"{self.nodes.iloc[i]['Type']}: " + " | ".join([m for m, c in eval(self.nodes.iloc[i]["mention_counts"]).items()])}) + '\n')
                        counts = eval(self.nodes.iloc[i]["mention_counts"])
                        total_count = sum(counts.values())
                        # max_count = max(counts.values())
                        m_list = []
                        for j, (m, c) in enumerate(sorted(counts.items(), key=lambda item:item[1], reverse=True)):
                            # if c < max_count * 0.1:
                            if len(m_list) >= 100:
                                break
                            # m_list += [f"{self.nodes.iloc[i]['Type']}: {m}"] * max(1, int(100 * c / total_count))
                            m_list += [f"{self.nodes.iloc[i]['Type']} {m} ({self.nodes.iloc[i]['ConceptID']})"] * max(1, int(100 * c / total_count))
                            # f.write(json.dumps({"id": f"{i}_{j}", "contents": f"{self.nodes.iloc[i]['Type']}: {m}"}) + '\n')
                        f.write(json.dumps({"id": f"{i}", "contents": ' | '.join(m_list)}) + '\n')
            # os.makedirs(index_dir, exist_ok=True)
            os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(mention_dir, self.index_dir))
        self.index = LuceneSearcher(os.path.join(self.index_dir))
        self.edges = pd.read_csv(os.path.join(self.load_dir, "edges_train.tsv"), sep='\t')
        # reorganize self.edges as {"x_id": pd.DataFrame} for fast retrieval
        self.x_id_dict = {x_id: group for x_id, group in self.edges.groupby('x_id')}
        self.node2mention = dict(self.nodes.apply(lambda row: (f"{row['Type']}|{row['ConceptID']}", row['mention']), axis=1).tolist())

    def retrieve_nodes(self, text: str, k=1):
        hits = self.index.search(text, k=k)
        # ids = [h.docid.split('_')[0] for h in hits]
        ids = [h.docid for h in hits]
        return self.nodes.iloc[ids].to_dict('records')

    def retrieve_edges(self, texts: list[str] | str | None = None, nodes: list[dict] | dict | None = None, max_n_edges: int = 32, max_depth: int = 2, max_width: int = -1, seed: int | None = 0):
        if nodes is None:
            assert texts is not None
            texts = [texts] if type(texts) == str else texts
            nodes = [self.retrieve_nodes(text) for text in texts]
            nodes = [n[0] for n in nodes if len(n) > 0]
        
        nodes = [nodes] if type(nodes) == dict else nodes

        if len(nodes) == 0:
            return []
        elif len(nodes) == 1:
            return_edges = self.edges[(self.edges["x_id"] == f"{nodes[0]['Type']}|{nodes[0]['ConceptID']}") + (self.edges["y_id"] == f"{nodes[0]['Type']}|{nodes[0]['ConceptID']}")].to_dict('records')
            # return_edges = self.edges[f"{nodes[0]['Type']}|{nodes[0]['ConceptID']}"]
            return_edges = [[e] for e in return_edges]
        else:
            return_edges = []
            for n in nodes:
                source_id = f"{n['Type']}|{n['ConceptID']}"
                target_ids = [f"{t['Type']}|{t['ConceptID']}" for t in nodes]
                if source_id not in self.x_id_dict:
                    continue
                sub_graph = self.x_id_dict[source_id]
                reach_target = sub_graph["y_id"].isin(target_ids)
                return_edges += sub_graph[reach_target].to_dict("records")
                to_be_explored = [[step] for step in sub_graph[~reach_target].sample(n=min(max_width if max_width != -1 else (~reach_target).sum(), (~reach_target).sum()), random_state=seed).to_dict("records")]
                while len(to_be_explored):
                    curr_chain = to_be_explored.pop(0)
                    if len(curr_chain) == max_depth:
                        break
                    if curr_chain[-1]["y_id"] not in self.x_id_dict:
                        continue
                    sub_graph = self.x_id_dict[curr_chain[-1]["y_id"]]
                    reach_target = sub_graph["y_id"].isin(target_ids)
                    return_edges += [curr_chain + [step] for step in sub_graph[reach_target].to_dict("records")]
                    to_be_explored += [curr_chain + [step] for step in sub_graph[~reach_target].sample(n=min(max_width if max_width != -1 else (~reach_target).sum(), (~reach_target).sum()), random_state=seed).to_dict("records")]
            # retrieve relations for each node if no edge among them is found
            if len(return_edges) == 0:
                return_edges = [e for n in nodes for e in self.retrieve_edges(nodes=n, max_n_edges=max_n_edges//len(nodes), seed=seed)]

        if len(return_edges) > max_n_edges:
            return_edges = random.sample(return_edges, max_n_edges)
        
        return return_edges

    def edge2text_v1(self, edge):
        return f"{edge['x_id'].split('|')[0]} {self.node2mention[edge['x_id']]} has the relation \"{edge['relation']}\" with {edge['y_id'].split('|')[0]} {self.node2mention[edge['y_id']]}"
    
    def edge2text(self, edge):
        return f"{edge['x_id'].split('|')[0]} {self.node2mention[edge['x_id']]} ({edge['x_id'].split('|')[1]}) has the relation \"{edge['relation']}\" with {edge['y_id'].split('|')[0]} {self.node2mention[edge['y_id']]} ({edge['y_id'].split('|')[1]})"