import re
import nltk
import heapq

class SummarizeParagraph(object):
    """
        Class contains code for text summarization.
    """
    def __init__(self, n):
        self.n = n

    def preprocess(self, text):
         text = re.sub(r'\[[0-9]*\]', ' ', text)
         text = re.sub(r'\s+', ' ', text)
         formatted_text = re.sub('[^a-zA-Z]', ' ', text)
         formatted_text = re.sub(r'\s+', ' ', formatted_text)  
         return text, formatted_text

    def sent_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def summarize(self, text):
        stopwords = nltk.corpus.stopwords.words('english')
        text, preprocessed_text = self.preprocess(text)
        print preprocessed_text
        sentence_list = self.sent_tokenize(text)
        word_frequencies = {}  
        for word in nltk.word_tokenize(preprocessed_text):  
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequncy = max(word_frequencies.values())
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        
        sentence_scores = {}  
	for sent in sentence_list:  
	    for word in nltk.word_tokenize(sent.lower()):
		if word in word_frequencies.keys():
		    if len(sent.split(' ')) < 30:
		        if sent not in sentence_scores.keys():
		            sentence_scores[sent] = word_frequencies[word]
		        else:
		            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(self.n, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

summarize_para = SummarizeParagraph(3)
para = "Artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and other animals. Many tools are used in AI, including versions of search and mathematical optimization, artificial neural networks, and methods based on statistics, probability and economics. The traditional problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception and the ability to move and manipulate objects. When access to digital computers became possible in the middle 1950s, AI research began to explore the possibility that human intelligence could be reduced to symbol manipulation. One proposal to deal with this is to ensure that the first generally intelligent AI is 'Friendly AI', and will then be able to control subsequently developed AIs. Nowadays, the vast majority of current AI researchers work instead on tractable  applications (such as medical diagnosis or automobile navigation). Machine learning, a fundamental concept of AI research since the field's inception, is the study of computer algorithms that improve automatically through experience."
print(summarize_para.summarize(para))
