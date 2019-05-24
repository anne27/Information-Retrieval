import numpy as np
import os
import os.path
from os import path
from time import time
import operator
import nltk
from nltk.corpus import stopwords
import pickle
import string
from os.path import isfile, join
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from collections import OrderedDict
from operator import itemgetter
from natsort import natsorted
from nltk.stem import *
from nltk.corpus import stopwords
from collections import defaultdict
import math
from os import listdir
from os.path import isfile, join
from num2words import num2words
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
import matplotlib.pyplot as plt

def read_file(filename):
    with open(filename, 'r', encoding="ascii", errors="surrogateescape") as f:
        stuff=f.read()
    f.close()
    return stuff

def remove_punctuations(final_string, vocab, doc_counts, fileno):
    finallist=[]
    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(final_string)
    table = str.maketrans('', '', '\t')
    token_list = [word.translate(table) for word in token_list]
    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in token_list]
    token_list = [str for str in stripped_words if str]
    token_list = [word.lower() for word in token_list]
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for item in token_list:
            if item not in stop_words:
                item = stemmer.stem(item)
                vocab.add(item)
                finallist.append(item)
                if item not in doc_counts:          # If term not in dictionary.
                    doc_counts[item] = [1, fileno]  # Doc count is 1, current file is fileno.
                else:
                    if doc_counts[item][1] != fileno:   # If the same file is not repeating.
                        doc_counts[item][0] += 1        # Add 1 to doc count.
                    doc_counts[item][1] = fileno        # Update current file.         
    return finallist, vocab, doc_counts

def tf_calc(doc_info, freqDict_list):
    # Function to return the term frequency. Tf is frequency of term in doc /
    # Total no. of terms in the doc.

    TF = {}                                                 # List of tfs for doc.
    for document in freqDict_list:
        docID = document['doc_id']
        doc_size = doc_info[docID]['doc_length']
        term_freq_list = document['freq_dict']
        for curterm in term_freq_list:
            freq_curterm = term_freq_list[curterm]
            tf_curterm = freq_curterm / doc_size
            if docID in TF:
                # If the document already exists.
                TF[docID][curterm] = tf_curterm
            else:
                # Document not in the dict.
                TF[docID] = {}                              # Create a blank dict for this doc.
                TF[docID][curterm] = tf_curterm + 1
            #curdict = {'doc_id': docID, 'tf': tf_curterm, 'term': curterm}
            #TF.append(curdict)
    print('TF')
    return TF

def idf_calc(vocab, doc_counts, no_of_docs):
    IDF = {}
    for word in vocab:
        IDF[word] = math.log10(no_of_docs / doc_counts[word][0])
    print('IDF')
    return IDF
  
def find_TF_IDF(TF, IDF, vocab, no_of_docs):
    TF_IDF = {}
    # Iterate through the documents.
    for doc in range(no_of_docs):
        # Now iterate through the entire vocab for this doc.
        for word in vocab:
            # Find the tf of this (word, doc) pair.
            try:
                tf = TF[doc][word]
            except:
                tf = 0
            idf = IDF[word]
            
            if doc in TF_IDF:
                # If the document already exists.
                TF_IDF[doc][word] = tf*idf
            else:
                TF_IDF[doc] = {}
                TF_IDF[doc][word] = tf*idf
    print('TF_IDF')
    return TF_IDF

def find_TF_IDF_query_specific(TF, IDF, word, doc):
    # Find the tf of this (word, doc) pair.
    try:
        tf = TF[doc][word]
    except:
        tf = 0
    idf = IDF[word]
    tfidf = tf*idf       
    return tfidf

def return_top_k(query_vec, doc_vec_dict):  
    doc_sims = []
    
    for i in range(no_of_docs):
         # Get the doc vec for this document.
        dv = doc_vec_dict[i]
        # Compare the qv with document vectors.
        doc_sims.append(cosine_similarity([dv], [qv]))

    # Return query results.
    top_k_doc_ids = sorted(range(len(doc_sims)), key=lambda kk: doc_sims[kk], reverse=True)
    top_k_docs = [doc_mapping[i] for i in top_k_doc_ids][:k]
    return top_k_docs

def get_centroid(rel_doc_vecs):
    total_vecs = len(rel_doc_vecs)
    sum_vec = [sum(i) for i in zip(*rel_doc_vecs)]
    sum_vec = [i/total_vecs for i in sum_vec]
    return sum_vec

folder_names = natsorted(os.listdir("20_newsgroups_mini"))
stop_words = set(stopwords.words('english')) 

if path.exists('files/tf.npy') and path.exists('files/idf.npy') and path.exists('files/doc_mapping.npy') and path.exists('files/vocab.npy'):
	TF = np.load('files/tf.npy').item()
	IDF = np.load('files/idf.npy').item()
	doc_mapping = np.load('files/doc_mapping.npy').item()   # DocID to filename.
	vocab = np.load('files/vocab.npy')          # Load total vocabulary.

else:
	#### First open the folder and read all the files. ####
	mypath = "20_newsgroups_mini"
	subfolders = natsorted(os.listdir(mypath))
	doc_mapping = {}
	fileno = 0                                                  # Map the doc name to a single integer (ID).
	doc_info = []                                               # List of dicts {docID: 1, doc_length: 11}.
	freqDict_list = []                                          # List of dicts {'doc_id':1. 'freq_dict':{term:f}}
	vocab = set()
	doc_counts = {}

	# Now map these files to docIDs and read them individually.

	for folder_name in subfolders:
	    onlyfiles = natsorted(os.listdir(mypath + "/" + folder_name))
	    for file_name in onlyfiles:
		doc_mapping[fileno] = folder_name + "/" + file_name
		stuff = read_file(mypath + "/" + folder_name + "/" + file_name)
		temp = remove_punctuations(stuff, vocab, doc_counts, fileno)
		final_token_list = temp[0]          # This is the list of words in order of the text.
		vocab = temp[1]                     # Update vocab.
		doc_counts = temp[2]
		doc_info.append({'doc_id':fileno , 'doc_length':len(final_token_list)})
		
		# Now create a frequency dict out of this list.
		fq = defaultdict( int )
		for w in final_token_list:
		    fq[w] += 1

		# Append to the freqDict_list.
		freqDict_list.append({'doc_id':fileno, 'freq_dict': fq})
		
		fileno += 1                                 # Increment the file no. counter for document ID mapping

	no_of_docs = fileno
	vocab = list(vocab)
	np.save('files/doc_mapping.npy', doc_mapping)
	np.save('files/vocab.npy', vocab)

	TF = tf_calc(doc_info, freqDict_list)
	IDF = idf_calc(vocab, doc_counts, no_of_docs)

	np.save('files/tf.npy', TF)
	np.save('files/idf.npy', IDF)

no_of_docs = len(doc_mapping.keys())        # Total no. of docs.

# Accept query and k.
print("Enter query")
my_query = input()
k=10                                        # Top 10 docs.

# Query processing.
temp = set()
q_terms = remove_punctuations(my_query, temp, {}, 0)[0]
modified_q_terms = []
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

for word in q_terms:
        # Check if not stopword.
        if word not in stop_words:
            if word.isdigit():                              # number correction
                word = num2words(int(word))
            word = stemmer.stem(word)                       # Stem the word
            modified_q_terms.append(word)


# Get the top docs according to the vector space notation.
doc_sims = []

# For similarity based score retrieval.
qf = defaultdict( int )                         # Now create a frequency dict out of this query list.
for w in modified_q_terms:
    qf[w] += 1

qv = []                                         # Create query vector.
for i in range (len(vocab)):
    try:
        word = vocab[i]
    except:
        word = ''        
    # Find the frequency of this word in the query.
    try:
        qv.append(qf[word])
    except:
        qv.append(0)

if path.exists('files/document_vectors.npy'):
	doc_vec_dict = np.load('files/document_vectors.npy').item()

else:
	# Creation of doc_vec_dict.
	doc_vec_dict = {}
	for i in range(no_of_docs):
	    print(i)
	    doc_name = doc_mapping[i]    
	    # Create the doc vec for this document.
	    dv = []
	    for w in range (len(vocab)):
		try:
		    word = vocab[w]
		    tfidf_body = find_TF_IDF_query_specific(TF, IDF, word, i)       # TFIDF for body.
		    dv.append(tfidf_body)
		except:
		    dv.append(0)                    # utf-32-le error.
	    doc_vec_dict[i] = dv
	    doc_sims.append(cosine_similarity([dv], [qv]))
	np.save('files/document_vectors.npy', doc_vec_dict)

refine = 1
counter = 0
types = []
types.append('q'+str(counter))
query_vec_log = []                      # Query vectors log.
query_vec_log.append(qv)                # Append the initial unmodified query.

while (refine == 1):
    top_k_docs = return_top_k(qv, doc_vec_dict)     # Return top k docs.
    for i in range(10):
        print(i, top_k_docs[i])

    # Now provide user relevance for the ten docs retrieved.
    print("Select documents you think are relevant. Enter space separated numbers")
    rels = input()
    rels = rels.split()
    rel_docs = []
    rel_doc_vecs = []
    for i in range(len(rels)):
        rel_docs.append(top_k_docs[i])
        rel_doc_vecs.append(doc_vec_dict[i])

    # Get the centroid for these docs.
    cen_vec = get_centroid(rel_doc_vecs)

    # Update the query vector qv.
    alpha = 1
    beta = 0.75
    gamma = 0.15
    
    l1 = [alpha*x for x in qv]
    l2 = [beta*x for x in cen_vec]
    qv = [l1[i]+l2[i] for i in range(len(l1))]      # Modified query vector.
    query_vec_log.append(qv)                        # Append new qv to log.

    counter += 1
    types.append('q'+str(counter))
    
    qv_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(query_vec_log)

    # Plot qv embedded.
    for i,typee in enumerate(types):
        x = qv_embedded[:, 0][i]
        y = qv_embedded[:, 1][i]
        plt.scatter(x, y, marker='x')
        plt.text(x+0.3, y+0.3, typee, fontsize=9)
    
    #plt.scatter(, , marker="x")
    plt.show()
    
    print("Enter 0 to continue, 1 to quit")
    opt = int(input())
    if opt == 1:
        break
