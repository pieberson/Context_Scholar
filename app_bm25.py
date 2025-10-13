from flask import Flask, render_template, request
from collections import defaultdict, Counter
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_app import create_app, db
from rank_bm25 import BM25Okapi
import pandas as pd
import math
import re
import numpy as np
import os, json

app = create_app()

# --- Load CSV Files ---
print("Loading corpus, queries, and qrels...")

corpus_df = pd.read_csv("corpus_complete.csv")
queries_df = pd.read_csv("queries.csv")
qrels_df = pd.read_csv("qrels.csv")

corpus = {}
for _, row in corpus_df.iterrows():
    doc_id = str(row["doc_id"])
    title = str(row.get("title", "Untitled"))
    text = str(row.get("text", ""))
    citations = int(row.get("citations", 0) or 0)
    year = int(float(row.get("year", 0))) if not pd.isna(row.get("year")) else "N/A"
    authors_raw = row.get("authors", "")
    paper_url = str(row.get("paper_url", "")).strip()
    authors = authors_raw.strip('"').strip("'") if isinstance(authors_raw, str) else ""

    corpus[doc_id] = {
        "title": title,
        "text": text,
        "citations": citations,
        "year": year,
        "authors": authors,
        "paper_url": paper_url
    }

queries = {
    str(row["query_id"]): row["text"]
    for _, row in queries_df.iterrows()
}

qrels = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query_id"])][row["doc_id"]] = int(row["score"])


# --- Text preprocessing ---
_token_re = re.compile(r"[^\w\s]")

def tokenize(text: str):
    if text is None:
        return []
    text = text.lower()
    text = _token_re.sub("", text)
    return text.split()


# --- BM25 Indexing (Using rank_bm25) ---
print("Building BM25 index with rank_bm25...")

doc_id_list = list(corpus.keys())
doc_texts = [
    (str(corpus[doc_id].get("title", "")) + ". " + str(corpus[doc_id].get("text", ""))).strip()
    for doc_id in doc_id_list
]

tokenized_corpus = [tokenize(doc) for doc in doc_texts]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index built for {len(tokenized_corpus)} documents.")


# --- Metrics ---
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

def compute_nfairr_citation(ranked_doc_ids, top_k=100):
    """
    NFaiRR fairness metric for citation balance.
    Higher = top results favor low-cited papers.
    """
    top_ranked = ranked_doc_ids[:top_k]
    if not top_ranked:
        return 0

    citation_values = [corpus[doc_id].get("citations", 0) for doc_id in top_ranked]
    norm_citations = [math.log1p(c + 1e-6) for c in citation_values]
    max_cite = max(norm_citations) if norm_citations else 1
    fairness_weights = [1 - (c / max_cite) for c in norm_citations]

    discounted_fairness = [
        w / math.log2(rank + 2) for rank, w in enumerate(fairness_weights)
    ]

    actual_score = sum(discounted_fairness)
    ideal_sorted = sorted(fairness_weights, reverse=True)
    ideal_score = sum(w / math.log2(rank + 2) for rank, w in enumerate(ideal_sorted))

    return actual_score / ideal_score if ideal_score != 0 else 0


# --- BM25 Search using library ---
def search_local(query_text, top_k=100):
    query_tokens = tokenize(query_text)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    ranked_doc_ids = [doc_id_list[i] for i in ranked_indices]
    ranked_scores = [scores[i] for i in ranked_indices]

    ranked = list(zip(ranked_doc_ids, ranked_scores))
    bm25_lookup = {doc_id_list[i]: scores[i] for i in ranked_indices}
    return ranked, bm25_lookup


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', queries=queries)


@app.route('/results', methods=['GET', 'POST'])
def results():
    experiment_mode = request.args.get("experiment_mode") or request.form.get("experiment_mode", "off")
    query_id = request.args.get("query_id") or request.form.get("query_id")
    query_text = request.args.get("query") or request.form.get("query")

    if experiment_mode == "on":
        selected_query_id = query_id
        if not selected_query_id:
            return render_template('results.html', query="", results=[], queries=queries, experiment_mode=experiment_mode)
        query_text = queries[selected_query_id]
    elif not query_text:
        return render_template('results.html', query="", results=[], queries=queries, experiment_mode=experiment_mode)

    # --- Run BM25 retrieval ---
    ranked, bm25_lookup = search_local(query_text, top_k=100)

    rel_map = {}
    if experiment_mode == "on":
        rel_map = qrels.get(query_id, {})

    predicted_rels, final_results, ranked_doc_ids = [], [], []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        doc = corpus[doc_id]
        rel_score = rel_map.get(doc_id, 0) if experiment_mode == "on" else 0
        predicted_rels.append(rel_score)
        ranked_doc_ids.append(doc_id)

        final_results.append({
            'rank': rank,
            'title': doc['title'],
            'abstract': doc['text'],
            'url': doc.get("paper_url") or f"https://www.semanticscholar.org/paper/{doc_id}",
            'authors': doc.get("authors", "Unknown"),
            'year': doc.get("year", "N/A"),
            'score': round(float(score), 4),
            'citations': doc.get("citations", 0)
        })

    # --- Metrics ---
    if experiment_mode == "on" and predicted_rels:
        ideal_rels = sorted([rel_map.get(doc_id, 0) for doc_id in ranked_doc_ids], reverse=True)
        ndcg = compute_ndcg(predicted_rels, ideal_rels)
        mrr = compute_mrr(predicted_rels)
    else:
        ndcg, mrr = 0, 0

    nfairr = compute_nfairr_citation(ranked_doc_ids, top_k=100)

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
    with app.app_context():
        db.create_all()
        print("Database tables created successfully.")
        print("Flask app is starting (BM25 via rank_bm25)...")
    app.run(debug=True, port=5002)
