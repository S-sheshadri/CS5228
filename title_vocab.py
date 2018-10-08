"""
    Vocabulary builder from titles column.
"""
import re
import pickle
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

def read_titles(title_path):
    titles = []
    with open(title_path, "r") as fp:
        titles = fp.read().lower().strip().split("\n")
    return titles

def read_news_articles():
    num_files = 4
    data_str = ''
    for num_file in range(num_files):
        file_path = "Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            data_str += fp.read()
    lines = data_str.split("##")
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
    print(preliminary_count)
    print(representation)
    print(len(representation))
    return representation

def create_bow(stemmed_tokens):
    texts = []
    for id, lines in stemmed_tokens:
        texts += lines
    dictionary = corpora.Dictionary(texts)
    dictionary.save('dictionary.dict')
    
    size_of_dictionary = len(dictionary)
    #Get the doc 2 bag-of-words model
    bow_representation = []
    for id, lines in stemmed_tokens:
        para = []
        for line in lines:
            para += line
        bow_representation.append((id, get_representation(dictionary.doc2bow(para), size_of_dictionary)))
    return bow_representation


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


lines = read_news_articles()
#path = "titles"
#lines = read_titles(path)
processed_lines = map(lambda tup: (tup[0], preprocess(tup[1])) if tup[1] != 'Empty' else tup, lines)
tokenized_lines = map(lambda tup: (tup[0], tokenize(tup[1])), processed_lines)
stopword_removed_lines = map(lambda tup: (tup[0], remove_stopwords(tup[1])), tokenized_lines)
#tagged_docs = tag_docs(stopword_removed_lines)
#print(tagged_docs)
stemmed_tokens = map(lambda tup: (tup[0], stem(tup[1])), stopword_removed_lines)
bow = create_bow(stemmed_tokens)

#print stopword_removed_titles
#model = create_doc_two_vec(tagged_docs)
#model.save("save/trained.model")
#model.save_word2vec_format('save/trained.word2vec')
