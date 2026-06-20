# Embeddings

Embeddings are dense vector representations that encode semantic meaning into fixed-size arrays of numbers. They are the bridge between human-readable text and the mathematical operations that power search, retrieval, clustering, and generation in modern AI systems.

## Foundations: Word2Vec and GloVe

The embedding revolution started with Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014). Word2Vec uses predictive models (Skip-gram, CBOW) to learn word vectors from local context. GloVe factorizes global co-occurrence statistics. Both produce vectors where semantic relationships become geometric ones — the famous example: king - man + woman ≈ queen.

Jay Alammar's "Illustrated Word2Vec" provides one of the best visual introductions to how these methods work.

## Sentence Embeddings

Word embeddings represent individual words, but most applications need to compare sentences or documents. Sentence-BERT (Reimers & Gurevych, 2019) solved this by using siamese BERT networks with mean pooling, reducing a 65-hour pairwise similarity computation (10K sentences) to 5 seconds. This created the Sentence Transformers library, now the most-used embedding library.

Modern sentence embedding models include **E5** (Microsoft, contrastive pre-training on 1B text pairs with query/passage prefixes), **GTE** (Alibaba, multi-stage contrastive learning, outperforms OpenAI embeddings at 10× fewer parameters), and OpenAI's **text-embedding-3** models.

## Matryoshka Representation Learning

MRL (Kusupati et al., NeurIPS 2022) is an elegant technique that trains embeddings to be useful at any prefix length — like Russian nesting dolls. The most important information is encoded in the first dimensions. This means you can truncate a 1536-dim embedding to 256 dims and still get useful results, enabling up to 14× storage savings and 14× faster retrieval. MRL powers OpenAI's text-embedding-3 models.

## Evaluation: MTEB

The Massive Text Embedding Benchmark (Muennighoff et al., 2023) standardized embedding evaluation across 8 task types, 58 datasets, and 112 languages. Its key finding: no single embedding model dominates all tasks. The best model depends on your specific use case — retrieval-optimized embeddings may underperform on classification, and vice versa.

## How Embeddings Connect to RAG

Embeddings are the foundation of [[retrieval-augmented-generation]]. Documents are chunked, embedded, and stored in a vector database. At query time, the query is embedded and the most similar document chunks are retrieved via vector similarity (typically cosine similarity or dot product). The quality of your embeddings directly determines the quality of your retrieval.

## Related Topics
- [[retrieval-augmented-generation]] — Embeddings power the retrieval step in RAG
- [[transformer-architecture]] — Modern embedding models are transformer-based
- [[inference-optimization]] — Embedding dimension reduction trades quality for speed

## Sources
- Word2Vec / GloVe foundational papers
- Sentence-BERT (arxiv:1908.10084)
- E5 (arxiv:2212.03533)
- Matryoshka Representation Learning (arxiv:2205.13147)
- MTEB Benchmark (arxiv:2210.07316)
- The Illustrated Word2Vec — Jay Alammar
