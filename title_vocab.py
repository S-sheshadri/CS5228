"""
    Vocabulary builder from titles column.
"""
import re
import pickle
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

def read_titles(title_path):
    titles = []
    with open(title_path, "r") as fp:
        titles = fp.read().lower().strip().split("\n")
    return titles


def preprocess(titles):
    processed_titles = []
    for title in titles:
        processed_title = re.sub(r'\W+', ' ', title).strip()
        processed_title = re.sub(r'\w*\d\w*', '', processed_title).strip()
        processed_titles.append(processed_title)
    return processed_titles
 
def tokenize(titles):
    tokenized_titles = []
    for title in titles:
        tokenized_titles.append(word_tokenize(title))
    return tokenized_titles

def remove_stopwords(tokenized_titles):
    titles = []
    stop_words = stopwords.words("english")
    for title in tokenized_titles:
        titles.append([word for word in title if word not in stop_words])
    return titles

def stem(tokens_list):
    stemmed_list = []
    p_stemmer = PorterStemmer()
    for token_list in tokens_list:
        stemmed_list.append([p_stemmer.stem(i) for i in token_list])
    return stemmed_list

def tag_docs(tokens_list):
    tagged_docs = []
    for index, token_list in enumerate(tokens_list):
        tagged_docs.append(TaggedDocument(token_list, str(index)))
    return tagged_docs

def create_doc_two_vec(docs):
    model = gensim.models.Doc2Vec(docs,
            size=20,
            window=3,
            min_count=0,
            workers=10,
            alpha=0.025,
            min_alpha=0.025)
    model.train(docs, total_examples=len(docs), epochs=200)
    return model

path = "titles"
titles = read_titles(path)
processed_titles = preprocess(titles)
tokenized_titles = tokenize(processed_titles)
stopword_removed_titles = remove_stopwords(tokenized_titles)
tagged_docs = tag_docs(stopword_removed_titles)
#stemmed_tokens = stem(stopword_removed_titles)
print stopword_removed_titles
model = create_doc_two_vec(tagged_docs)
model.save("save/trained.model")
model.save_word2vec_format('save/trained.word2vec')
