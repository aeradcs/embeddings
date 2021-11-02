import pandas as pd
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

import utils


def save_bow(filename, name):
    df = pd.read_csv(filename)
    df['news'] = df['news'].apply(lambda text: utils.preprocessing(text))
    print("df after preprocessing\n", df)

    dictionary = Dictionary(df['news'])
    print(dictionary)
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
    for v in tf_idf_matrix:
        print(v)
    corpora.MmCorpus.serialize(f'embeddings_dfs/tf_idf_matrix_{name}.mm', tf_idf_matrix)


def load_tf_idf(name):
    dictionary = Dictionary.load(f"embeddings_dfs/tf_idf_dictionary_{name}.dict")
    print(dictionary)
    corpus = list(corpora.MmCorpus(f"embeddings_dfs/tf_idf_corpus_{name}.mm"))
    print(corpus)
    tf_idf_matrix = list(corpora.MmCorpus(f"embeddings_dfs/tf_idf_matrix_{name}.mm"))
    print(tf_idf_matrix)


if __name__ == '__main__':
    # save_bow("marked_dfs/test.csv", "test")
    # load_bow("test")

    save_tf_idf("marked_dfs/test.csv", "test")
    load_tf_idf("test")

    # bow("marked_dfs/bloomberg_marked_df.csv")
    # bow("marked_dfs/business_standart_marked_df.csv")
    # bow("marked_dfs/yahoo_finance_marked_df.csv")

