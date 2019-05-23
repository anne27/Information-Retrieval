# Indexing

We use the 20 Newsgroups dataset to build a unigram inverted index.
Further, support is provided for the following boolean commands:
- x OR y
- x AND y
- x AND NOT y
- x OR NOT y
Finally, support for searching for phrase queries using Positional Indexes. For simplicity, this index is built only on comp.graphics and rec.motorcycles folders.

## Description

Unigram inverted index is built on the entire 20 newsgroups dataset, in the form of a dictionary. The format of the inverted index is <br/>
`{term: [total_frequency, [list of docs containing term]]}`<br/>
For example,\
`inv_index['graph']` is\
`[75, [1054, 1139, 1214, 1214, 1214, 1215, 1215, 1245, 1288, 1444, 1530, 1616,
1616, 1681, 1690, 1690, 1690, 1691, 1691, 1718, 1771, 1984, 1992, 1992, 2329,
2333, 3171, 3674, 3708, 5003, 5016, 5027, 5027, 5038, 5038, 5038, 5116, 5116,
5230, 5366, 5433, 5473, 5641, 5660, 5703, 5779, 5779, 5779, 5791, 5791, 5791,
5900, 5909, 5923, 5952, 6967, 7054, 7481, 11102, 11445, 13644, 13848, 14074,
14074, 14405, 14521, 14521, 14554, 14554, 14680, 14739, 16208, 16897,
18535, 18779]]`<br/>

Here, `inv_index` of the term ‘graph’ contains a list, with the first term as 75, which is the overall no. of documents containing the term, and a list containing the document IDs of the term.

Preprocessing steps:
- Documents are tokenized using NLTK’s TweetTokenizer.
- Punctuations are removed.
- Headers and footers are removed.
- Tokens are converted to lowercase.
- Stop words are removed during the creation of inverted index, but NOT removed for positional index.
- Tokens are stemmed using PorterStemmer.<br/>
Stop words are not removed since they can appear adjacently in a phrasal query, e.g. “*To be or not to be*”.  

Positional index is built on the comp.graphics and rec.motorcycles folders, in the form of a dictionary. The format of the positional index is  
`{term: [total_frequency, [list of docs containing term: [positions of the term in that
document]]}`\
For example,\
`pos_index['motor']` is\
`[32, {121: [76], 1112: [15], 1131: [95], 1150: [192, 205], 1153: [75, 189], 1158: [130], 1228: [51],
1237: [218], 1282: [138], 1285: [224], 1297: [18], 1326: [61], 1330: [29], 1380: [51], 1393: [62],
1430: [29], 1433: [169, 172], 1577: [41], 1605: [81], 1691: [206], 1708: [112], 1720: [187], 1765:
[35], 1776: [212], 1779: [49], 1913: [24], 1914: [238], 1964: [287], 1976: [122]}]`

## Running the Code

### Inverted Index
Run `inverted_index.py`.

### Positional Index
Run `positional_index.py`.

## Sample Queries

### Inverted Index

**Input**\
x AND y | science AND auto

**Output**\
`['rec.autos/102900', 'talk.politics.mideast/76266', 'rec.autos/102910', 'comp.graphics/38852',
'comp.graphics/38821', 'comp.graphics/38375']`

**Input**\
x OR y | sci OR pyjama

**Output**\
`['sci.electronics/53543', 'sci.med/59602', 'sci.med/59094', 'misc.forsale/76434',
'talk.politics.mideast/75970', 'talk.politics.guns/53319', 'sci.med/59488', 'comp.graphics/38851',
'comp.graphics/38724', 'sci.crypt/15178', 'talk.politics.guns/53325', 'comp.graphics/38988',
'sci.med/59499', 'alt.atheism/53654', 'comp.graphics/38618', 'sci.med/59127','comp.graphics/38495', 'sci.med/59518', 'misc.forsale/75929', 'sci.med/58568', 'sci.med/58578',
'comp.graphics/38377', 'sci.space/59913', 'comp.sys.mac.hardware/52298', 'sci.med/58764',
'comp.graphics/38257', 'sci.med/58767', 'sci.crypt/15998', 'sci.med/59158',
'comp.graphics/38778', 'sci.med/58776', 'talk.politics.mideast/76053', 'misc.forsale/75955',
'sci.med/59548', 'sci.med/59165', 'sci.space/61201', 'soc.religion.christian/20929',
'comp.os.ms-windows.misc/10190', 'sci.med/59183', 'sci.space/61091', 'sci.space/62399',
'sci.med/58800', 'sci.crypt/15900', 'sci.med/59188', 'comp.sys.mac.hardware/52342',
'sci.med/59576', 'sci.med/59323', 'sci.space/62414', 'sci.med/59077', 'sci.med/59207']`

### Positional Index

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
