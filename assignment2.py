import json
import os
import string
import math
import sys
import argparse
from collections import defaultdict

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# If you haven't downloaded NLTK resources, uncomment the lines below and run once:
# nltk.download('stopwords')
# nltk.download('punkt')

def tokenize_and_remove_punctuations(text):
    """
    Remove punctuation and digits, then tokenize.
    """
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)
    text_no_digits = ''.join(ch for ch in text_no_punct if not ch.isdigit())
    tokens = wordpunct_tokenize(text_no_digits.lower())
    return tokens

def get_stopwords():
    """
    Return the set of English stopwords from NLTK.
    """
    return set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    """
    Tokenize, remove stopwords, filter out words with length < 3, then apply stemming.
    Return the list of processed tokens.
    """
    tokens = tokenize_and_remove_punctuations(text)
    stopwords = get_stopwords()
    filtered_tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def read_corpus_with_raw(corpus_file):
    """
    Read corpus.jsonl, build a dict doc_id -> {"tokens": [...], "raw": full_text}.
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            content = doc.get('title', '') + ' ' + doc.get('text', '')
            tokens = preprocess_text(content)
            corpus[doc_id] = {"tokens": tokens, "raw": content}
    return corpus

def get_relevance(relevance_file):
    """
    Read relevance TSV, format: qid unused docid rel
    Only store rel>0 if needed, but adjust as necessary.
    Returns dict: query_id -> [doc_id, ...]
    """
    from collections import defaultdict
    relevances = defaultdict(list)
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                relevances[qid].append(docid)
    return relevances

def read_queries_with_raw(queries_file, valid_query_ids):
    """
    Read queries.jsonl, only keep queries that appear in valid_query_ids.
    Returns dict: query_id -> {"tokens": [...], "raw": query text}
    """
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            q_id = q['_id']
            if q_id in valid_query_ids:
                raw_text = q.get('text', '')
                tokens = preprocess_text(raw_text)
                queries[q_id] = {"tokens": tokens, "raw": raw_text}
    return queries

def build_inverted_index(corpus):
    """
    Build an inverted index: term -> set(doc_id).
    """
    from collections import defaultdict
    inverted_index = defaultdict(set)
    for doc_id, data in corpus.items():
        tokens = data["tokens"]
        for token in set(tokens):
            inverted_index[token].add(doc_id)
    return inverted_index

# ---------------- TF-IDF BASELINE ---------------- #

def calculate_tf(tokens):
    from collections import defaultdict
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def calculate_idf(corpus):
    """
    Compute IDF: idf[term] = log(N / df).
    """
    from collections import defaultdict
    idf = {}
    N = len(corpus)
    term_doc_count = defaultdict(int)
    for data in corpus.values():
        for term in set(data["tokens"]):
            term_doc_count[term] += 1
    for term, df in term_doc_count.items():
        idf[term] = math.log(N / df) if df > 0 else 0
    return idf

def calculate_tfidf(tf, idf):
    tfidf = {}
    for term, freq in tf.items():
        tfidf[term] = freq * idf.get(term, 0)
    return tfidf

def cosine_similarity(query_vec, doc_vec):
    dot_product = 0.0
    for term, weight in query_vec.items():
        if term in doc_vec:
            dot_product += weight * doc_vec[term]
    query_norm = math.sqrt(sum(weight**2 for weight in query_vec.values()))
    doc_norm = math.sqrt(sum(weight**2 for weight in doc_vec.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot_product / (query_norm * doc_norm)

def baseline_retrieval(queries, corpus, inverted_index, idf, top_k=100):
    """
    TF-IDF retrieval with cosine similarity. Return top_k docs.
    """
    results = {}
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_tokens = queries[query_id]["tokens"]
        query_tf = calculate_tf(query_tokens)
        query_tfidf = calculate_tfidf(query_tf, idf)

        # Candidate documents via inverted index
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs.update(inverted_index.get(token, set()))

        scores = {}
        for doc_id in candidate_docs:
            doc_tf = calculate_tf(corpus[doc_id]["tokens"])
            doc_tfidf = calculate_tfidf(doc_tf, idf)
            score = cosine_similarity(query_tfidf, doc_tfidf)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results[query_id] = sorted_docs[:top_k]
    return results

# ---------------- ADVANCED NEURAL METHODS (No TF usage) ---------------- #

def dense_retrieval(queries, corpus, top_k=100, model_name="all-MiniLM-L6-v2"):
    """
    Sentence-BERT Dense Retrieval.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("Loading dense model:", model_name)
    model = SentenceTransformer(model_name)

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id]["raw"] for doc_id in doc_ids]
    print("Computing document embeddings...")
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)

    results = {}
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        query_embedding = model.encode(query_text, convert_to_numpy=True)

        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            cosine_scores = np.zeros(len(doc_embeddings))
        else:
            cosine_scores = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)

        top_indices = np.argsort(-cosine_scores)[:top_k]
        top_candidates = [(doc_ids[i], float(cosine_scores[i])) for i in top_indices]
        results[query_id] = top_candidates
    return results

def bert_rerank(queries, corpus, baseline_results, top_k=100,
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Cross-Encoder BERT Re-ranking on top_k TF-IDF (or other) candidates.
    """
    from sentence_transformers import CrossEncoder
    import numpy as np

    print("Loading cross-encoder model:", model_name)
    model = CrossEncoder(model_name)

    results = {}
    for query_id in sorted(baseline_results, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        candidates = baseline_results[query_id]  # list of (doc_id, score)
        candidate_pairs = []
        candidate_doc_ids = []
        for doc_id, _ in candidates:
            doc_text = corpus[doc_id]["raw"]
            candidate_pairs.append([query_text, doc_text])
            candidate_doc_ids.append(doc_id)

        print(f"Re-ranking query {query_id} with {len(candidate_pairs)} candidates...")
        scores = model.predict(candidate_pairs)  # returns relevance scores
        candidate_score_pairs = list(zip(candidate_doc_ids, scores))
        candidate_score_pairs.sort(key=lambda x: x[1], reverse=True)
        results[query_id] = candidate_score_pairs[:top_k]
    return results

# ---------------- Universal Sentence Encoder (Hugging Face) ---------------- #

def use_retrieval(queries, corpus, top_k=100, model_name="sentence-transformers/use-cmlm-multilingual"):
    """
    Retrieve documents using the Universal Sentence Encoder from sentence-transformers on Hugging Face.
    This model does NOT require TensorFlow. It uses a PyTorch-backed version of USE.
    
    Steps:
      1. Load the model via SentenceTransformer(model_name).
      2. Encode all documents.
      3. Encode each query.
      4. Rank by cosine similarity.

    Requirements:
      pip install sentence-transformers numpy
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("Loading USE model (PyTorch wrapper):", model_name)
    model = SentenceTransformer(model_name)

    # Prepare doc IDs and texts
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id]["raw"] for doc_id in doc_ids]

    print("Computing document embeddings (USE, PyTorch-based)...")
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)

    results = {}
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        query_embedding = model.encode(query_text, convert_to_numpy=True)

        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            cosine_scores = np.zeros(len(doc_embeddings))
        else:
            cosine_scores = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)

        top_indices = np.argsort(-cosine_scores)[:top_k]
        top_candidates = [(doc_ids[i], float(cosine_scores[i])) for i in top_indices]
        results[query_id] = top_candidates
    return results

# ---------------- MS MARCO Implementation ---------------- #

def msmarco_distilbert_rerank(queries, corpus, baseline_results, top_k=100, 
                             model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Re-ranks documents using MS MARCO MiniLM cross-encoder.
    This model is specifically designed for relevance ranking in information retrieval.
    
    Steps:
      1. Take base results from another retrieval method
      2. Use cross-encoder to re-rank the top candidates
      3. Return the re-ranked results
    
    Requirements:
      pip install sentence-transformers
    """
    from sentence_transformers import CrossEncoder
    
    print(f"Loading MS MARCO cross-encoder for re-ranking: {model_name}")
    model = CrossEncoder(model_name, max_length=512)
    
    results = {}
    for query_id in sorted(baseline_results, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        candidates = baseline_results[query_id]  # list of (doc_id, score)
        
        # Prepare query-document pairs for the cross-encoder
        candidate_pairs = []
        candidate_doc_ids = []
        
        for doc_id, _ in candidates:
            doc_text = corpus[doc_id]["raw"]
            candidate_pairs.append([query_text, doc_text])
            candidate_doc_ids.append(doc_id)
        
        print(f"Re-ranking query {query_id} with MS MARCO cross-encoder, {len(candidate_pairs)} candidates...")
        
        # Get relevance scores from the cross-encoder
        relevance_scores = model.predict(candidate_pairs)
        
        # Create (doc_id, score) pairs and sort by score
        doc_score_pairs = list(zip(candidate_doc_ids, relevance_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top_k results
        results[query_id] = doc_score_pairs[:top_k]
    
    return results

# ---------------- OUTPUT in TREC FORMAT ---------------- #

def write_results(results, output_filename, run_tag):
    """
    Write top_k results for each query in TREC format:
      query_id Q0 doc_id rank score tag
    """
    with open(output_filename, "w", encoding="utf-8") as out_file:
        for query_id in sorted(results, key=lambda x: int(x)):
            rank = 1
            for doc_id, score in results[query_id]:
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n")
                rank += 1
    print(f"Results written to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Neural Retrieval Methods (Including MS MARCO DistilBERT)")
    parser.add_argument("corpus_file", help="Path to corpus.jsonl")
    parser.add_argument("queries_file", help="Path to queries.jsonl")
    parser.add_argument("relevance_file", help="Path to test.tsv (relevance judgments)")
    parser.add_argument("--method", choices=["tfidf", "dense", "bert", "use", "msmarco-rerank"],
                        default="tfidf",
                        help="Retrieval method: tfidf, dense, bert, use, msmarco-rerank")
    parser.add_argument("--dense_model", default="all-MiniLM-L6-v2",
                        help="Model name for SentenceTransformer (Dense Retrieval)")
    parser.add_argument("--bert_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Model name for CrossEncoder (BERT Re-ranking)")
    parser.add_argument("--use_model", default="sentence-transformers/universal-sentence-encoder",
                        help="Model name for the Hugging Face USE variant (no TF required)")
    parser.add_argument("--msmarco_rerank_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Model name for MS MARCO re-ranking")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of documents to return per query")
    args = parser.parse_args()

    print("Reading relevance judgments...")
    relevances = get_relevance(args.relevance_file)
    valid_query_ids = set(relevances.keys())
    print(f"{len(valid_query_ids)} queries found in relevance file.")

    print("Reading and preprocessing corpus...")
    corpus = read_corpus_with_raw(args.corpus_file)
    print(f"Corpus contains {len(corpus)} documents.")

    print("Reading and preprocessing queries...")
    queries = read_queries_with_raw(args.queries_file, valid_query_ids)
    print(f"{len(queries)} queries to process.")

    run_tag = "MyIRsystem"

    if args.method == "tfidf":
        print("Building inverted index...")
        inverted_index = build_inverted_index(corpus)
        print("Calculating IDF...")
        idf = calculate_idf(corpus)
        print("Running TF-IDF retrieval (baseline)...")
        results = baseline_retrieval(queries, corpus, inverted_index, idf, top_k=args.top_k)
        output_file = "Results_tfidf.txt"

    elif args.method == "dense":
        print("Running Dense Retrieval (Sentence-BERT)...")
        results = dense_retrieval(queries, corpus, top_k=args.top_k, model_name=args.dense_model)
        output_file = "Results_dense.txt"

    elif args.method == "bert":
        print("Building inverted index for candidate generation...")
        inverted_index = build_inverted_index(corpus)
        print("Calculating IDF...")
        idf = calculate_idf(corpus)
        print("Running baseline TF-IDF retrieval to get candidate docs...")
        baseline_results = baseline_retrieval(queries, corpus, inverted_index, idf, top_k=args.top_k)
        print("Re-ranking candidates with CrossEncoder...")
        results = bert_rerank(queries, corpus, baseline_results, top_k=args.top_k,
                              model_name=args.bert_model)
        output_file = "Results_bert.txt"

    elif args.method == "use":
        print("Running Universal Sentence Encoder retrieval (Hugging Face model)...")
        results = use_retrieval(queries, corpus, top_k=args.top_k, model_name=args.use_model)
        output_file = "Results_USE.txt"
        
    elif args.method == "msmarco-rerank":
        print("Building inverted index for candidate generation...")
        inverted_index = build_inverted_index(corpus)
        print("Calculating IDF...")
        idf = calculate_idf(corpus)
        print("Running baseline TF-IDF retrieval to get candidate docs...")
        baseline_results = baseline_retrieval(queries, corpus, inverted_index, idf, top_k=args.top_k)
        print("Re-ranking candidates with MS MARCO Cross-Encoder...")
        results = msmarco_distilbert_rerank(queries, corpus, baseline_results, top_k=args.top_k,
                                          model_name=args.msmarco_rerank_model)
        output_file = "Results_msmarco_rerank.txt"

    else:
        print("Invalid retrieval method selected.")
        sys.exit(1)

    # Write final results in TREC format
    write_results(results, output_file, run_tag)
    print("Retrieval completed. Best results saved to:", output_file)

if __name__ == "__main__":
    main()