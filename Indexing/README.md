# Indexing

We use the 20 Newsgroups dataset to build a unigram inverted index.
Further, support is provided for the following boolean commands:
- x OR y
- x AND y
- x AND NOT y
- x OR NOT y
<br/>
Finally, support for searching for phrase queries using Positional Indexes. For simplicity, this index is built only on comp.graphics and rec.motorcycles folders.

## Description

Preprocessing steps:
- Documents are tokenized using NLTK’s TweetTokenizer.
- Punctuations are removed.
- Headers and footers are removed.
- Tokens are converted to lowercase.
- Stop words are NOT removed.
- Tokens are stemmed using PorterStemmer.
Stop words are not removed since they can appear adjacently in a query, e.g. “To be or not to be”.

## Running the Code

### Inverted Index
Run `inverted_index.py`.
### Positional Index
Run `positional_index.py`.

## Sample Queries
**Input**<br/>
Enter query<br/>
*will be sent*

**Output**<br/>
Document Name [Position of last word]<br/>
comp.graphics/37261 [224]<br/>
comp.graphics/38377 [5044]<br/>
comp.graphics/38609 [225]<br/>
comp.graphics/38851 [5148]<br/>
comp.graphics/38920 [215]<br/>
comp.graphics/38971 [133]

**Input**<br/>
Enter query<br/>
*submission deadline is*

**Output**<br/>
Document Name [Position of last word]<br/>
comp.graphics/37261 [215]<br/>
comp.graphics/38609 [216]
