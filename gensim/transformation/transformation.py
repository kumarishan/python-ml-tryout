import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('../tmp/deerwester.dict')
corpus_bow = corpora.MmCorpus('../tmp/deerwester.mm')
print "#" * 5, 'corpus MM bow', "#" * 5
print corpus_bow

def print_doc_in_corpus(corpus, name):
    print
    print "#" * 5, 'corpus ', name, ' docs', "#" * 5
    for doc in corpus:
        print doc
    print
    print

# Tf-Idf Transformation model
tfidf = models.TfidfModel(corpus_bow) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus_bow]
print_doc_in_corpus(corpus_tfidf,'tf-idf')

corpora.MmCorpus.serialize('../tmp/corpus_tfidf.mm', corpus_tfidf)

# tranforming Tf-idf via latent semantic indexing into a latent 2-D space
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow -> tfifd -> fold-in-lsi

print "Topics in LSI/LSA"
print lsi.show_topics(2)
print_doc_in_corpus(corpus_lsi, 'LSI')

lsi.save('../tmp/model.lsi')
lsi = models.LsiModel.load('../tmp/model.lsi')

# Error with schpy fblas...
# Random Projection,RP
# rp = models.RpModel(corpus_tfidf, num_topics=2)
# corpus_rp = rp[corpus_tfidf]
# print_doc_in_corpus(corpus_rp, 'RP')

# Latent Dirichlet Allocation
lda = models.LdaModel(corpus_bow, id2word=dictionary, num_topics=2)
corpus_lda = lda[corpus_bow]

print "Topics in lda"
print lda.show_topics(-1)
print_doc_in_corpus(corpus_lda, 'LDA')

# Heirarchical Dirichlet Process
hdp = models.HdpModel(corpus=corpus_bow, id2word=dictionary)
print hdp.print_topics(topics=20, topn=10)