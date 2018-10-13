"""
    Vocabulary builder from titles column.
"""
import re
import numpy as np
import pickle
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

def read_titles():
    titles = []
    with open("titles", "r") as fp:
        for line in fp:
            try:
                id, sent = line.lower().split("|")
            except:
                splitted = line.lower().split("|")
                id,sent = splitted[0], ' '.join(splitted[1:])
                
            titles.append((id, [sent.strip()]))
    return titles

def read_news_articles():
    num_files = 4
    lines = []
    for num_file in range(num_files):
        file_path = "test_data/Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("##")
    lines = map(lambda line: line.split("|"), lines)
    lines =  filter(lambda tup: len(tup) == 2, lines)
    lines = filter(lambda tup: tup[1].strip() != 'Empty', lines)
    lines = map(lambda tup: (tup[0], tup[1].lower().split("\n")), lines)
    return lines

def preprocess(lines):
    processed_lines = []
    for line in lines:
        processed_line = re.sub(r'\W+', ' ', line).strip()
        processed_line = re.sub(r'\w*\d\w*', '', processed_line).strip()
        if processed_line:
            processed_lines.append(processed_line)
    return processed_lines
 
def tokenize(lines):
    tokenized_lines = []
    for line in lines:
        tokenized_lines.append(word_tokenize(line))
    return tokenized_lines

def remove_stopwords(tokenized_lines):
    lines = []
    stop_words = stopwords.words("english")
    for line in tokenized_lines:
        lines.append([word for word in line if word not in stop_words])
    return lines

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

def get_representation(preliminary_count, size):
    representation = []
    for idx, count in preliminary_count:
        if len(representation) == idx:
            representation.append(count)
        else:
            while len(representation) < idx:
                representation.append(0)
            representation.append(count)

    while len(representation) < size:
        representation.append(0)
    return representation

def create_bow(stemmed_tokens):
    texts = []
    for id, lines in stemmed_tokens:
        texts += lines
    print texts
    dictionary = corpora.Dictionary(texts)
    dictionary.save('dictionary.dict')
    
    size_of_dictionary = len(dictionary)
    #Get the doc 2 bag-of-words model
    bow_representation = []
    id_list = []
    for id, lines in stemmed_tokens:
        para = []
        for line in lines:
            para += line
        id_list.append(id)
        bow_representation.append(get_representation(dictionary.doc2bow(para), size_of_dictionary))
    return np.array(id_list), np.array(bow_representation)


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

def complete_preprocessing(lines):
    processed_lines = map(lambda tup: (tup[0], preprocess(tup[1])) if tup[1] != 'Empty' else tup, lines)
    tokenized_lines = map(lambda tup: (tup[0], tokenize(tup[1])), processed_lines)
    stopword_removed_lines = map(lambda tup: (tup[0], remove_stopwords(tup[1])), tokenized_lines)
    stemmed_tokens = map(lambda tup: (tup[0], stem(tup[1])), stopword_removed_lines)
    return stemmed_tokens

def augment_with_title(lines, all_lines):
    out = []
    curr_id = 0
    for id, text in lines:
        if id == str(curr_id):
            out.append((id, text))
        else:
            while str(curr_id) != id:
                out.append(all_lines[curr_id])
                curr_id += 1
            out.append((id, text))
        curr_id += 1
    return out

#print("Reading articles")
#articles = read_news_articles()
"""
print("Reading titles")
titles = read_titles()
#print("Processing articles")
#prepreprocessed_articles = complete_preprocessing(articles)
print("processing titles")
preprocessed_titles = complete_preprocessing(titles)
print("augmenting with titles")
#full_representation = augment_with_title(prepreprocessed_articles, preprocessed_titles)
#print(full_representation)
id, bow = create_bow(preprocessed_titles)
print(id.shape, bow.shape)
np.save("ids", id)
np.save("bow_titles", bow)

"""

#print stopword_removed_titles
#model = create_doc_two_vec(tagged_docs)
#model.save("save/trained.model")
#model.save_word2vec_format('save/trained.word2vec')
