import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora,models, similarities

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove comman words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]

print texts

# assigning ids to each token
dictionary = corpora.Dictionary(texts)
dictionary.save('../tmp/deerwester.dict')
print dictionary
print dictionary.token2id

# to convert tokenized documents to vector
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('../tmp/deerwester.mm', corpus) # store to disk, for later use
corpora.SvmLightCorpus.serialize('../tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('../tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('../tmp/corpus.low', corpus)
print corpus
# print corpus: load it entirely into memory
print list(corpus) # calling list() will convert any sequence to a plain Python list
# or
# another way: print one document at a time, making use of streaming interface
for doc in corpus:
    print doc

# creating dictionary without loading files to memory
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in idsequence after words that were removed
print dictionary
