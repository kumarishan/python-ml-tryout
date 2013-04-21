import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('../tmp/deerwester.dict')
corpus_bow = corpora.MmCorpus('../tmp/deerwester.mm')

lsi = models.LsiModel(corpus_bow, id2word=dictionary, num_topics=2)

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
print vec_bow

vec_lsi = lsi[vec_bow]
print vec_lsi

index = similarities.MatrixSimilarity(lsi[corpus_bow])
index.save('../tmp/deerwester.index')

sims = index[vec_lsi] # perform a similarity query against the corpus bow
print list(enumerate(sims)) # print document number and document similarity

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sims