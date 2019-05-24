# Query OptimizationQ

We use Tf-Idf based vector space document retrieval to get top 10 documents based on a cosine similarity between query and document vector. Further, we provide a client for a user to give Relevance Feedback (tell which all docs are relevant and which are irrelevant). \

We optimize the query using Rocchio's Algorithm, and keep doing this process of taking user feedback and updating the query vector until the user quits the program.\
For simplicity, we use only `comp.graphics` and `rec.motorcycles` documents.

## Description

The vector space model implemented in the previous assignment (containing vector representations of documents and query vectors) is used to compute similarity between query and docs, and retrieve top-k documents. The user is asked to select relevant docs, and the query vector is modified through Rocchio’s algorithm.\

`q(modified) = alpha*q(original) + beta*centroid(relevant) - gamma*centroid(non-relevant)`\
alpha = 1\
beta = 0.75\
gamma = 0.15

## Running the Code
Run `rocchio.py`.

## Sample Queries

Enter query\
i love computer more than software\
0 comp.graphics/38305\
1 comp.graphics/39022\
2 comp.graphics/38659\
3 rec.motorcycles/104529\
4 comp.graphics/38776\
5 comp.graphics/38976\
6 comp.graphics/39045\
7 comp.graphics/38817\
8 comp.graphics/38848\
9 comp.graphics/38962\
Select documents you think are relevant. Enter space separated numbers\
0 1 6 7 9\
[t-SNE] Computing 1 nearest neighbors...\
[t-SNE] Indexed 2 samples in 0.000s...\
[t-SNE] Computed neighbors for 2 samples in 0.001s...\
[t-SNE] Computed conditional probabilities for sample 2 / 2\
[t-SNE] Mean sigma: 1125899906842624.000000\
[t-SNE] Computed conditional probabilities in 0.015s\
[t-SNE] Iteration 50: error = 29.8188801, gradient norm = 0.0060926 (50 iterations in 0.007s)\
[t-SNE] Iteration 100: error = 29.8188801, gradient norm = 0.0121347 (50 iterations in 0.012s)\
[t-SNE] Iteration 150: error = 29.8188801, gradient norm = 0.0104368 (50 iterations in 0.012s)\
[t-SNE] Iteration 200: error = 29.8188801, gradient norm = 0.0155561 (50 iterations in 0.012s)\
[t-SNE] Iteration 250: error = 29.8188801, gradient norm = 0.0065845 (50 iterations in 0.013s)\
[t-SNE] KL divergence after 250 iterations with early exaggeration: 29.818880\
[t-SNE] Iteration 300: error = 0.0000000, gradient norm = 0.0000000 (50 iterations in 0.008s)\
[t-SNE] Iteration 300: gradient norm 0.000000. Finished.\
[t-SNE] KL divergence after 300 iterations: 0.000000\
Enter 0 to continue, 1 to quit\
0\
0 comp.graphics/38305\
1 comp.graphics/39022\
2 comp.graphics/40062\
3 comp.graphics/38733\
4 comp.graphics/39062\
5 comp.graphics/38283\
6 comp.graphics/37960\
7 comp.graphics/37941\
8 comp.graphics/38659\
9 comp.graphics/38823\
Select documents you think are relevant. Enter space separated numbers\
0 2 3 5 6\
[t-SNE] Computing 2 nearest neighbors...\
[t-SNE] Indexed 3 samples in 0.000s...\
[t-SNE] Computed neighbors for 3 samples in 0.001s...\
[t-SNE] Computed conditional probabilities for sample 3 / 3\
[t-SNE] Mean sigma: 1125899906842624.000000\
[t-SNE] Computed conditional probabilities in 0.010s\
[t-SNE] Iteration 50: error = 45.4713135, gradient norm = 0.0229603 (50 iterations in 0.007s)\
[t-SNE] Iteration 100: error = 36.2256279, gradient norm = 0.0770725 (50 iterations in 0.013s)\
[t-SNE] Iteration 150: error = 31.8558121, gradient norm = 0.1628436 (50 iterations in 0.012s)\
[t-SNE] Iteration 200: error = 33.8701859, gradient norm = 0.0762896 (50 iterations in 0.013s)\
[t-SNE] Iteration 250: error = 33.9695320, gradient norm = 0.4137620 (50 iterations in 0.012s)\
[t-SNE] KL divergence after 250 iterations with early exaggeration: 33.969532\
[t-SNE] Iteration 300: error = 0.0000336, gradient norm = 0.0001135 (50 iterations in 0.006s)\
[t-SNE] Iteration 350: error = -0.0000000, gradient norm = 0.0000017 (50 iterations in 0.013s)\
[t-SNE] Iteration 400: error = -0.0000000, gradient norm = 0.0000000 (50 iterations in 0.013s)\
[t-SNE] Iteration 400: gradient norm 0.000000. Finished.\
[t-SNE] KL divergence after 400 iterations: -0.000000\
Enter 0 to continue, 1 to quit\
1
