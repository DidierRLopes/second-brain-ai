# Word2Vec and GloVe: Foundational Word Embeddings

**Sources:** 
- Word2Vec: https://arxiv.org/abs/1301.3781 (Mikolov et al., Google, 2013)
- GloVe: https://nlp.stanford.edu/projects/glove/ (Pennington, Socher, Manning, Stanford, 2014)

## Key Concepts

Word2Vec (2013) and GloVe are the foundational embedding methods that launched the modern era of representation learning. Word2Vec uses predictive models (Skip-gram and CBOW) to learn word embeddings from local context windows. GloVe uses count-based matrix factorization of global word co-occurrence statistics.

Both produce dense vector representations where semantically similar words are close in vector space. The famous example: king - man + woman ≈ queen. These methods are essential background for understanding modern sentence and document embeddings.
