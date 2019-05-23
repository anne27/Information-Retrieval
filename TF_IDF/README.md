# TF-IDF Based Document Retrieval

This is a CLI tool for:
- **Tf-Idf based document retrieval**: For each query, the system will output top k documents based on tf-idf-matching-score.
- **Tf-Idf based vector space document retrieval**: For each query, the system will output top k documents based on a cosine similarity between query and document vector.

In addition, we give special attention to the terms in the document title.

## Description

Pre-processing of documents:
- Tokenized using NLTK’s TweetTokenizer.
- Punctuations removed.
- Stopwords removed.
- Remaining tokens are stemmed using NLTK’s PorterStemmer.
The same is done for the queries.

### Tf-Idf based document retrieval
Three dictionaries are maintained:
- **TF dictionary**: Given a term and a docID, this dictionary returns the term frequency.
- **IDF dictionary**: Given a term, it returns its inverse document frequency.
- Document Name to DocID.

The query is tokenized, cleaned and stemmed. Finally, the TF IDF value corresponding to each query term is calculated for each document, and the resulting TF IDF values for that document are summed up.\
The final TF IDF scores for all documents are stored in a list, and the top k scores are returned.

### Tf-Idf based vector space document retrieval

**Creation of query vector**:\
A vector of |vocab| size is created, with each value corresponding to the frequency of that word in the query.\
**Creation of doc vector**:\
A vector of |vocab| size is created, with each value corresponding to the TF-IDF score of that word-document pair.\

Cosine similarity between these 2 vectors is calculated, and this is repeated for all documents. Top k scores are returned.

### Handling the Titles
Titles are given more importance than the body because they are more information-rich and concise. TF-IDF scores are calculated for all the titles.\
`Score = 0.4*TF-IDF(body) + 0.6*TF-IDF(title)`

### Handling the Numbers in Queries
Numbers are converted into their word expansions using num2words.

## Running the Code
Run `title.py` to build the dictionaries for TF-IDF values in the title.
Run `retrieve_docs.py`.

## Sample Queries

**Input**\
Query: master young

**Output**\
Algo 1\
`['jim.asc', 'pussboot.txt', 'clevdonk.txt', 'advsayed.txt', 'horsdonk.txt', 'encamp01.txt', 'hareleph.txt', 'bluebrd.txt', 'lmermaid.txt', 'lgoldbrd.txt']`\
Algo 2\
`['jim.asc', 'quot', 'lgoldbrd.txt', 'horsdonk.txt', 'clevdonk.txt', 'pussboot.txt', 'SRE/index.html', 'timem.hac', 'glimpse1.txt', 'encamp01.txt']`

**Input**\
Query: 3 little pigs

**Output**\
Algo 1\
`['bullove.txt', 'lmtchgrl.txt', 'goldfish.txt', 'write', 'quarter.c6', 'lpeargrl.txt', 'blossom.pom', '3lpigs.txt', '3wishes.txt', '3student.txt']`\
Algo 2\
`['3lpigs.txt', 'bullove.txt', '3wishes.txt', 'lrrhood.txt', 'goldfish.txt', 'lgoldbrd.txt', 'quarter.c6', 'korea.s', 'timem.hac', 's&m_plot']`

**Input**\
Query: earth is dreamland

**Output**\
Algo 1\
`['peace.fun', 'quickfix', 'quarter.c15', 'nigel.2', 'narciss.txt', 'charlie.txt', 'spam.key', 'deathmrs.d', 'empsjowk.txt', 'dicegame.txt']`\
Algo 2\
`['quickfix', 'quarter.c15', 'cum', 'ghost', 'timem.hac', 'deathmrs.d', 'spam.key', 'redragon.txt', 'bullove.txt', 'space.txt']`

**Input**\
Query: All summer long, they roamed through the woods and over the plains,playing games and having fun.\

**Output**\
Algo 1\
`['game.txt', 'SRE/.desc', 'SRE/.head', 'SRE/sre01.tx', 'SRE/.musing', 'SRE/index.html', 'sqzply.txt', 'discocanbefun.txt', 'disco.be.fun', 'dicegame.txt']`\
Algo 2\
`['game.txt', '3lpigs.txt', 'korea.s', 'lrrhood.txt', 's&m_that', 'imonly17.txt', 'jim.asc', 'toilet.s', 'timem.hac', 'SRE/.desc']`
