from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
from collections import defaultdict, Counter
import pandas as pd
import torch
import math
import re
import numpy as np
import os, json
import torch.nn as nn

app = Flask(__name__)

# --- Custom WeightedPooling class  ---
class WeightedPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(WeightedPooling, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        attention_mask = features.get('attention_mask', None)

        scores = self.attention(token_embeddings).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        sentence_embeddings = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)
        features['sentence_embedding'] = sentence_embeddings
        return features

    def get_config_dict(self):
        return {'embedding_dim': self.embedding_dim}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f)

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        inst = cls(cfg['embedding_dim'])
        weights_path = os.path.join(input_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            inst.load_state_dict(state_dict)
        return inst


# --- Load CSV Files ---
print("Loading corpus, queries, and qrels...")

corpus_df = pd.read_csv("corpus_with_citations.csv")
queries_df = pd.read_csv("queries.csv")
qrels_df = pd.read_csv("qrels.csv")

corpus = {
    row["doc_id"]: {
        "title": row["title"],
        "text": row["text"],
        "citations": int(row.get("citations", 0))
    }
    for _, row in corpus_df.iterrows()
}

queries = {
    str(row["query_id"]): row["text"]
    for _, row in queries_df.iterrows()
}

qrels = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query_id"])][row["doc_id"]] = int(row["score"])


# --- Text preprocessing / tokenization utility ---
_token_re = re.compile(r"[^\w\s]")

def tokenize(text: str):
    if text is None:
        return []
    text = text.lower()
    text = _token_re.sub("", text)
    return text.split()


# --- Manual indexing ---
print("Building internal index...")

doc_id_list = list(corpus.keys())
doc_texts = [
    (str(corpus[doc_id].get("title") or "") + ". " + str(corpus[doc_id].get("text") or "")).strip()
    for doc_id in doc_id_list
]

tokenized_docs = [tokenize(t) for t in doc_texts]
N = len(tokenized_docs)
doc_lengths = [len(d) for d in tokenized_docs]
avgdl = sum(doc_lengths) / N if N > 0 else 0.0

k1 = 1.5
b = 0.75

term_freqs = []
doc_freq = defaultdict(int)
inverted_index = defaultdict(set)

for doc_idx, tokens in enumerate(tokenized_docs):
    tf = Counter(tokens)
    term_freqs.append(tf)
    for term in tf.keys():
        doc_freq[term] += 1
        inverted_index[term].add(doc_idx)

idf = {}
for term, df in doc_freq.items():
    idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

print(f"Indexed {N} documents, avgdl={avgdl:.2f}, vocabulary={len(doc_freq)} terms")


# --- Load Models ---
print("Loading models...")
MODEL1_OUT = "./biencoder_minilm_weighted_msmarco-1"
MODEL2_OUT = "./crossencoder_citation_trec_covid-1"

# bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
bi_encoder = SentenceTransformer(MODEL1_OUT)  # loads custom WeightedPooling automatically
cross_encoder = CrossEncoder(MODEL2_OUT)


# --- Evaluation Metrics ---
def compute_dcg(relevances):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))

def compute_ndcg(predicted_rels, ideal_rels):
    dcg = compute_dcg(predicted_rels)
    idcg = compute_dcg(sorted(ideal_rels, reverse=True))
    return dcg / idcg if idcg != 0 else 0

def compute_mrr(relevances):
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1 / (i + 1)
    return 0

def compute_nfairr_citation(ranked_doc_ids, top_k=50):
    """
    NFaiRR fairness metric for citation counts.
    Evaluates whether the top-k results are dominated by highly cited papers.
    Higher score = more balance (not just high-cited docs at the top).
    """

    # Consider only top_k documents
    top_ranked = ranked_doc_ids[:top_k]

    citation_weights = []
    for rank, doc_id in enumerate(top_ranked):
        citations = corpus[doc_id].get("citations", 1)
        # fairness weight: lower for highly cited docs, higher for low-cited
        fairness_weight = 1 / math.log1p(citations)
        score = fairness_weight / (rank + 1)  # rank discount
        citation_weights.append(score)

    # Ideal: sort by fairness (low citations first → highest weight)
    ideal_order = sorted(
        [1 / math.log1p(corpus[doc_id].get("citations", 1)) for doc_id in top_ranked],
        reverse=True
    )
    ideal_scores = [w / (i + 1) for i, w in enumerate(ideal_order)]

    actual = sum(citation_weights)
    ideal = sum(ideal_scores)
    return actual / ideal if ideal != 0 else 0



# --- Initial manual scoring ---
def raw_scores(query_tokens):
    scores = [0.0] * N
    if N == 0:
        return scores

    for term in query_tokens:
        if term not in inverted_index:
            continue
        term_idf = idf.get(term, 0.0)
        for doc_idx in inverted_index[term]:
            tf = term_freqs[doc_idx].get(term, 0)
            dl = doc_lengths[doc_idx]
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            score_contrib = term_idf * ((tf * (k1 + 1)) / denom) if denom > 0 else 0.0
            scores[doc_idx] += score_contrib
    return scores

def initial_scores(query_tokens):
    scores = [0.0] * N
    if N == 0:
        return scores

    for term in query_tokens:
        if term not in inverted_index:
            continue
        term_idf = idf.get(term, 0.0)
        for doc_idx in inverted_index[term]:
            tf = term_freqs[doc_idx].get(term, 0)
            dl = doc_lengths[doc_idx]
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            score_contrib = term_idf * ((tf * (k1 + 1)) / denom) if denom > 0 else 0.0
            scores[doc_idx] += score_contrib

    # Calculate adaptive thresholds using the median
    if scores:
        bm25_threshold = np.median(scores)  # Use median of current scores
    else:
        bm25_threshold = 0.0
    
    citations_list = [corpus[doc_id_list[doc_idx]].get("citations", 0) for doc_idx in range(N)]
    if citations_list:
        citation_threshold = np.median(citations_list)  # Use median of all citations
    else:
        citation_threshold = 0.0
    
    print(f"Calculated Thresholds: BM25 = {bm25_threshold:.2f}, Citations = {citation_threshold:.0f}")

    # 🔹 Apply citation boost with conditional logic
    for doc_idx in range(N):
        bm25_score = scores[doc_idx]
        citations = corpus[doc_id_list[doc_idx]].get("citations", 0)

        # Check for both conditions before applying the boost
        if bm25_score > bm25_threshold and citations < citation_threshold:
            boost = 1 + 0.1 * math.log1p(citations)
            scores[doc_idx] *= boost

    return scores

def search_local(query_text, top_k=50, bm25_k=200, bi_k=100):
    query_tokens = tokenize(query_text)

    # Step 1: Initial retrieval
    raw_score = raw_scores(query_tokens) 
    raw_bm25_scores = initial_scores(query_tokens)
    bm25_top_indices = sorted(range(len(raw_bm25_scores)),
                              key=lambda i: raw_bm25_scores[i],
                              reverse=True)[:bm25_k]

    bm25_top_doc_ids = [doc_id_list[i] for i in bm25_top_indices]
    bm25_top_texts = [doc_texts[i] for i in bm25_top_indices]

    if not bm25_top_texts:
        return []

    # Step 2: Bi-Encoder reranking
    # Pass only the top 100 from BM25 to the bi-encoder
    num_for_biencoder = min(bi_k, len(bm25_top_texts))
    bm25_for_biencoder_ids = bm25_top_doc_ids[:num_for_biencoder]
    bm25_for_biencoder_texts = bm25_top_texts[:num_for_biencoder]
    
    query_embedding = bi_encoder.encode(query_text, convert_to_tensor=True)
    doc_embeddings = bi_encoder.encode(bm25_for_biencoder_texts, convert_to_tensor=True)
    bi_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0].tolist()

    # Sort the bi-encoder scores to get the top `top_k` for the cross-encoder
    bi_top_indices = sorted(range(len(bi_scores)),
                            key=lambda i: bi_scores[i],
                            reverse=True)[:top_k]

    bi_top_doc_ids = [bm25_for_biencoder_ids[i] for i in bi_top_indices]
    bi_top_texts = [bm25_for_biencoder_texts[i] for i in bi_top_indices]
    bi_top_scores = [bi_scores[i] for i in bi_top_indices]

    # Step 3: Cross-Encoder reranking
    # The top 50 from the bi-encoder are already in bi_top_texts
    cross_inputs = [(query_text, doc_text) for doc_text in bi_top_texts]
    cross_scores = cross_encoder.predict(cross_inputs)

    # Final sort of the top 50 documents from the cross-encoder
    ranked = sorted(
        zip(bi_top_doc_ids, bi_top_scores, cross_scores),
        key=lambda x: x[2],  # sort by cross-encoder
        reverse=True
    )[:top_k]

    initial_raw_lookup = {doc_id_list[i]: raw_score[i] for i in bm25_top_indices}
    initial_lookup = {doc_id_list[i]: raw_bm25_scores[i] for i in bm25_top_indices}
    bi_lookup = {doc_id: score for doc_id, score in zip(bi_top_doc_ids, bi_top_scores)}

    return ranked, initial_lookup, initial_raw_lookup, bi_lookup

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', queries=queries)

@app.route('/results', methods=['GET', 'POST'])
def results():
    experiment_mode = request.args.get("experiment_mode") or request.form.get("experiment_mode", "off")

    if experiment_mode == "on":
        # --- Experiment mode ON: uses query_id dropdown ---
        selected_query_id = request.args.get('query_id') or request.form.get('query_id')
        if not selected_query_id:
            return render_template('results.html', query="", results=[], queries=queries, experiment_mode=experiment_mode)

        query_text = queries[selected_query_id]

    else:
        # --- Experiment mode OFF: uses free-text input ---
        query_text = request.args.get("query") or request.form.get("query")
        if not query_text:
            return render_template('results.html', query="", results=[], queries=queries, experiment_mode=experiment_mode)

    # --- Run retrieval pipeline ---
    ranked, initial_lookup, initial_raw_lookup,  bi_lookup = search_local(query_text, top_k=50)

    rel_map = {}
    if experiment_mode == "on":
        # Only qrels available for experiment mode
        selected_query_id = request.args.get('query_id') or request.form.get('query_id')
        rel_map = qrels.get(selected_query_id, {})

    predicted_rels = []
    final_results = []
    ranked_doc_ids = []

    value_threshold = np.median(list(initial_raw_lookup.values())) if initial_raw_lookup else 0.0

    for rank, (doc_id, bi_score, score) in enumerate(ranked, start=1):
        doc = corpus[doc_id]
        rel_score = rel_map.get(doc_id, 0) if experiment_mode == "on" else 0
        predicted_rels.append(rel_score)
        ranked_doc_ids.append(doc_id)

        raw_val = float(initial_raw_lookup.get(doc_id, 0.0))
        boosted_val = float(initial_lookup.get(doc_id, 0.0))

        fairness_boosted = not math.isclose(raw_val, boosted_val, rel_tol=1e-6)
        high_relevance = raw_val > value_threshold

        final_results.append({
            'rank': rank,
            'title': doc['title'],
            'abstract': doc['text'],
            'url': f"https://www.semanticscholar.org/paper/{doc_id}",
            'score': round(float(score), 4),
            'bi_score': round(float(bi_lookup.get(doc_id, 0.0)), 4),
            'initial_raw': round(raw_val, 4),
            'initial_score': round(boosted_val, 4),
            'doc_id': doc_id,
            'citations': doc.get("citations", 0),
            'fairness_boosted': fairness_boosted,
            'high_relevance': high_relevance
        })


    # --- Metrics ---
    if experiment_mode == "on":
        ideal_rels = sorted(rel_map.values(), reverse=True)
        ndcg = compute_ndcg(predicted_rels, ideal_rels)
        mrr = compute_mrr(predicted_rels)
    else:
        ndcg = 0
        mrr = 0

    nfairr = compute_nfairr_citation(ranked_doc_ids, top_k=50)

    return render_template(
        'results.html',
        query=query_text,
        results=final_results,
        ndcg=round(ndcg, 4),
        mrr=round(mrr, 4),
        nfairr=round(nfairr, 4),
        queries=queries,
        experiment_mode=experiment_mode
    )


# --- Run App ---
if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True)
