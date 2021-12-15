import pandas as pd
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import collections
import random

import utils


def save_bow(filename, name):
    df = pd.read_csv(filename)
    df['news'] = df['news'].apply(lambda text: utils.preprocessing(text))
    print("df after preprocessing\n", df)

    dictionary = Dictionary(df['news'])
    print(dictionary)
    # print(dictionary.token2id)
    dictionary.save(f"embeddings_dfs/bow_dictionary_{name}.dict")

    corpus = [dictionary.doc2bow(text) for text in df['news']]
    print(corpus)
    corpora.MmCorpus.serialize(f'embeddings_dfs/bow_corpus_{name}.mm', corpus)


def load_bow(name):
    dictionary = Dictionary.load(f"embeddings_dfs/bow_dictionary_{name}.dict")
    print(dictionary)
    corpus = list(corpora.MmCorpus(f"embeddings_dfs/bow_corpus_{name}.mm"))
    print(corpus)


def save_tf_idf(filename, name):
    df = pd.read_csv(filename)
    df['news'] = df['news'].apply(lambda text: utils.preprocessing(text))
    print("df after preprocessing\n", df)

    dictionary = Dictionary(df['news'])
    print(dictionary)
    dictionary.save(f"embeddings_dfs/tf_idf_dictionary_{name}.dict")

    corpus = [dictionary.doc2bow(text) for text in df['news']]
    print(corpus)
    corpora.MmCorpus.serialize(f'embeddings_dfs/tf_idf_corpus_{name}.mm', corpus)

    model = TfidfModel(corpus)
    tf_idf_matrix = [model[corpus[i]] for i in range(0, len(corpus))]
    # for v in tf_idf_matrix:
    #     print(v)
    print(tf_idf_matrix)
    corpora.MmCorpus.serialize(f'embeddings_dfs/tf_idf_matrix_{name}.mm', tf_idf_matrix)


def load_tf_idf(name):
    dictionary = Dictionary.load(f"embeddings_dfs/tf_idf_dictionary_{name}.dict")
    print(dictionary)
    corpus = list(corpora.MmCorpus(f"embeddings_dfs/tf_idf_corpus_{name}.mm"))
    print(corpus)
    tf_idf_matrix = list(corpora.MmCorpus(f"embeddings_dfs/tf_idf_matrix_{name}.mm"))
    print(tf_idf_matrix)


def doc2vec(filename, name):
    df = pd.read_csv(filename)
    df_orig = df.copy()
    df['news'] = df['news'].apply(lambda text: utils.light_preprocessing(text))
    print("df after preprocessing\n", df)
    df_test = pd.read_csv("marked_dfs/testtest.csv")
    df_test_orig = df_test.copy()
    df_test['news'] = df_test['news'].apply(lambda text: utils.light_preprocessing(text))
    print("df_test after preprocessing\n", df_test)

    train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['news'])]
    test_corpus = list(df_test['news'])

    model = Doc2Vec(vector_size=50, min_count=2, epochs=40, dm=1)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print(model.infer_vector(train_corpus[0][0]))

    vector = model.infer_vector(test_corpus[0])
    print("vector infer\n", vector)
    similar_doc = model.dv.most_similar(vector, topn=5)
    print("similar_doc\n", similar_doc)
    print("\n--------------------------------------------------------\n")
    print(f"FOR NEW = {df_test_orig.iloc[0]['news']} \nMOST SIMILAR IS {df_orig.iloc[similar_doc[0][0]]['news']}")


    # print("ASSESING==============================================================================")
    #
    # ranks = []
    # second_ranks = []
    # for doc_id in range(len(train_corpus)):
    #     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    #     sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    #     rank = [docid for docid, sim in sims].index(doc_id)
    #     ranks.append(rank)
    #
    #     second_ranks.append(sims[1])
    #
    # counter = collections.Counter(ranks)
    # print("counter", counter)
    #
    # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    #
    # doc_id = random.randint(0, len(train_corpus) - 1)
    #
    # # Compare and print the second-most-similar document
    # print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    # sim_id = second_ranks[doc_id]
    # print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))
    #
    # print("TESTING==============================================================================")
    # # Pick a random document from the test corpus and infer a vector from the model
    # doc_id = random.randint(0, len(test_corpus) - 1)
    # inferred_vector = model.infer_vector(test_corpus[doc_id])
    # sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    #
    # # Compare and print the most/median/least similar documents from the train corpus
    # print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


if __name__ == '__main__':
    # save_bow("marked_dfs/test1.csv", "test1")
    # load_bow("test1")

    # save_tf_idf("marked_dfs/test.csv", "test")
    # load_tf_idf("test")

    doc2vec("marked_dfs/test.csv", "test")

    # df = pd.DataFrame({0:[0,1,2,3], 1:['a', 'b', 'c', 'd']})
    # print(df)
    # print(df.iloc[3][1])

    # bow("marked_dfs/bloomberg_marked_df.csv")
    # bow("marked_dfs/business_standart_marked_df.csv")
    # bow("marked_dfs/yahoo_finance_marked_df.csv")

