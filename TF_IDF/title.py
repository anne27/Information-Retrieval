import numpy as np
import os
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
from sklearn.metrics.pairwise import cosine_similarity

def read_file(filename):
    with open(filename, 'r', encoding="ascii", errors="surrogateescape") as f:
        stuff=f.read()
    f.close()
    return stuff

def open_file(folder_path):
    final_string=""
    folder_path="stories"
    onlyfiles = [f for f in os.listdir(folder_path)]
    for f in onlyfiles:
        # Read the current file
        with open(folder_path+"/"+f,encoding="utf8", errors='ignore') as myfile:
            file_content=myfile.read()
            final_string+=file_content
    return final_string

def map_titles(sre = False):
    myfilename = 'files/titles'
    if (sre == True):
        myfilename = 'files/titles_sre'
    filename_to_title = {}                  # Filename to title mapping.
    with open(myfilename) as t:
        l = t.readlines()
    # The first line contains the filename and size.
    c = 1
    for line in l:
        line = line.replace('\n', '')
        if (c == 1):                        # Odd line.
            tokens = line.split('\t')
            file_name = tokens[0]
            c = 0
        else:                               # Even line.
            file_title = line
            filename_to_title[file_name] = file_title
            c = 1
    return filename_to_title
    
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

folder_names=natsorted(os.listdir("stories"))
stop_words = set(stopwords.words('english')) 

#### First open the folder and read ONLY files. ####
titles = map_titles()                       # Filename to title for docs.
titles_sre = map_titles(True)               # Filename to title for SRE docs.
mypath = "stories"
onlyfiles = natsorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
doc_mapping = {}
inv_doc_mapping = {}
fileno = 0                                                  # Map the doc name to a single integer (ID).
doc_info = []                                               # List of dicts {docID: 1, doc_length: 11}.
freqDict_list = []                                          # List of dicts {'doc_id':1. 'freq_dict':{term:f}}
vocab = set()
doc_counts = {}

# Now map these files to docIDs and read them individually.

for file_name in onlyfiles:
    doc_mapping[fileno] = file_name
    inv_doc_mapping[file_name] = fileno
    try:
        stuff = titles[file_name]
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
    except:
        continue

#### List ONLY subdirectories and handle these now. ####

onlydirs = ['stories/SRE']                       # Files from FARNON were removed.

# Now map these files to docIDs and read them individually.

for folder_name in onlydirs:
    file_names = natsorted(os.listdir(folder_name))
    for file_name in file_names:
        doc_mapping[fileno] = (folder_name + '/' + file_name).strip("stories/")
        inv_doc_mapping[(folder_name + '/' + file_name).strip("stories/")] = fileno
        try:
            stuff = titles_sre[file_name]
            temp = remove_punctuations(stuff, vocab, doc_counts, fileno)
            final_token_list = temp[0]              # This is the list of words in order of the text.
            vocab = temp[1]                         # Update the vocab after this tokenization.
            doc_counts = temp[2]
            doc_info.append({'doc_id':fileno , 'doc_length':len(final_token_list)})
            
            # Now create a frequency dict out of this list.
            fq = defaultdict( int )
            for w in final_token_list:
                fq[w] += 1

            # Append to the freqDict_list.
            freqDict_list.append({'doc_id':fileno, 'freq_dict': fq})
            
            fileno += 1                             # Increment the file no. counter for document ID mapping
        except:
            continue

no_of_docs = fileno
vocab = list(vocab)

if not os.path.exists('files_title'):
    os.makedirs('files_title')

np.save('files_title/doc_mapping.npy', doc_mapping)
np.save('files_title/inv_doc_mapping.npy', inv_doc_mapping)
np.save('files_title/vocab.npy', vocab)

TF = tf_calc(doc_info, freqDict_list)
IDF = idf_calc(vocab, doc_counts, no_of_docs)
TFIDF = find_TF_IDF(TF, IDF, vocab, no_of_docs)

np.save('files_title/tf.npy', TF)
np.save('files_title/idf.npy', IDF)
np.save('files_title/tfidf.npy', TFIDF)
