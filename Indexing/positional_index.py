import numpy as np
import os
import os.path
from os import path
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

def read_file(filename):
    with open(filename, 'r', encoding="ascii", errors="surrogateescape") as f:
        stuff=f.read()
    f.close()
    stuff=remove_header_footer(stuff)		# Remove header and footer.
    return stuff

def open_file():
	final_string=""
	folder_path="20_newsgroups/comp.graphics" #comp.graphics rec.motorcycles
	onlyfiles = [f for f in os.listdir(folder_path)]
	for f in onlyfiles:
		# Read the current file
		with open(folder_path+"/"+f,encoding="utf8", errors='ignore') as myfile:
			file_content=myfile.read()
			file_content=remove_header_footer(file_content)		# Remove header and footer.
			final_string+=file_content
	return final_string

def remove_header_footer(final_string):
	new_final_string=""
	tokens=final_string.split('\n\n')
	# Remove tokens[0] and tokens[-1]
	for token in tokens[1:-1]:
		new_final_string+=token+" "
	return new_final_string

def remove_punctuations(final_string):
	tokenizer=TweetTokenizer()
	token_list=tokenizer.tokenize(final_string)
	table = str.maketrans('', '', '\t')
	token_list = [word.translate(table) for word in token_list]
	punctuations = (string.punctuation).replace("'", "")
	trans_table = str.maketrans('', '', punctuations)
	stripped_words = [word.translate(trans_table) for word in token_list]
	token_list = [str for str in stripped_words if str]
	token_list=[word.lower() for word in token_list]
	return token_list

def or_command(x,y, inv_index):
    docs_with_x=[]
    docs_with_y=[]
    try:
        docs_with_x=inv_index[x][1]
        docs_with_y=inv_index[y][1]
    except:
        pass
    return list(set(docs_with_x+docs_with_y))           # Merge the lists
    

def and_command(x,y, inv_index):
    docs_with_x=[]
    docs_with_y=[]
    try:
        docs_with_x=inv_index[x][1]
        docs_with_y=inv_index[y][1]
    except:
        pass
    return list(set(docs_with_x) & set(docs_with_y))                        # Intersection of the lists

def not_command(x, inv_index, doc_mapping):
    docs_without_x=doc_mapping.values()                 # Take all the documents initially
    try:
        docs_with_x=inv_index[x]
    except:
        return docs_without_x                           # x doesnt exist in any document
    return [docid for docid in docs_without_x if docid not in docs_with_x]

def map_to_filename(doc_list, doc_mapping):
    final_doc_list=[]
    for doc in doc_list:
        final_doc_list.append(doc_mapping[doc])
    return final_doc_list

def match_for_two(doc_id1, doc_id2):
    match_dict={}

    # Find the common documents
    common_docids=sorted(list(doc_id1.keys() & doc_id2.keys()))

    for docid in common_docids:
        pos1=doc_id1[docid]
        pos2=doc_id2[docid]

        t1=0
        t2=0
        while (t1<len(pos1) and t2<len(pos2)):
            if (pos2[t2]-pos1[t1]==1):
                if docid not in match_dict:
                    match_dict[docid]=[pos2[t2]]
                else:
                    match_dict[docid].append(pos2[t2])
                t1+=1
                t2+=1
            elif (pos2[t2]-pos1[t1]>1):
                t1+=1
            else:
                t2+=1
                
    return match_dict


fileno=0                                                # Map the doc name to a single integer (ID).

stemmer = PorterStemmer()

if path.exists('pos_index.npy') and path.exists('pos_doc_mapping.npy'):
	pos_index=np.load('pos_index.npy').item()
	doc_mapping=np.load('pos_doc_mapping.npy').item()

## FORMING POSITIONAL INDEX ##
else:
	folder_names=["comp.graphics", "rec.motorcycles"]

	for folder_name in folder_names:
	    file_names=natsorted(os.listdir("20_newsgroups/"+folder_name))
	    print(folder_name)
	    for file_name in file_names:
		doc_mapping[fileno]=folder_name+"/"+file_name
		stuff=read_file("20_newsgroups/"+folder_name+"/"+file_name)
		final_token_list=remove_punctuations(stuff)	# This is the list of words in order of the text.
		for pos,term in enumerate(final_token_list):
		            # First stem the term
		            term=stemmer.stem(term)
		            if term in pos_index:
		                # Increment total freq by 1
		                pos_index[term][0]=pos_index[term][0]+1
		                # Check if the term has existed in that DocID before
		                if fileno in pos_index[term][1]:
		                    pos_index[term][1][fileno].append(pos)
		                else:
		                    pos_index[term][1][fileno]=[pos]
		            else:
		                pos_index[term]=[]              # Initialize the list
		                pos_index[term].append(1)       # The total frequency is 1
		                pos_index[term].append({})      # The postings list is initally empty
		                # Add doc ID to postings list
		                pos_index[term][1][fileno]=[pos]
		fileno+=1                                       # Increment the file no. counter for document ID mapping
	np.save('pos_index.npy', pos_index)
	np.save('pos_doc_mapping.npy', doc_mapping)

print("Enter query")
q=input()
q_list=q.split()

# Stem the query

for i in range(len(q_list)):
    q_list[i]=stemmer.stem(q_list[i])

# Handle the case of single word queries

if (len(q_list)==0):                # No query
    print("No query entered")
elif (len(q_list)==1):              # Single word query
    try:
        word1=q_list[0]
        # Retrieve its postings list
        pos_index1=pos_index[word1]
        doc_id1=pos_index1[1]
        print("Document Name [Position of last word]")
        for k, v in doc_id1.items():
            print(doc_mapping[k], v)
    except:
        print("Word not found in corpus.")
else:                               # Phrasal query
    word1=q_list[0]
    word2=q_list[1]

    # Find the postings list of the two words
    pos_index1=pos_index[word1]
    pos_index2=pos_index[word2]

    # Find the common document IDs between the two
    doc_id1=pos_index1[1]
    doc_id2=pos_index2[1]

    match_dict=match_for_two(doc_id1, doc_id2)

    for i in range(2,len(q_list)):
        pl2=pos_index[q_list[i]][1]
        match_dict=match_for_two(match_dict,pl2)
        
    print("Document Name [Position of last word]")
    for k, v in match_dict.items():
        print(doc_mapping[k], v)
