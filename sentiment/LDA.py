import csv
from nltk.stem.porter import *
from nltk.corpus import stopwords
from gensim.models import doc2vec, ldamodel
from gensim import corpora
from tqdm import tqdm

st = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    # stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split() if w not in st and w.isalpha()]
    return words


def LDA(origin_unlabeled_data):
    data = [preprocess_text(x) for x in tqdm(origin_unlabeled_data)]
    results = []
    # LDA
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    for num_top in [5]:
        print('the number of topic is:', num_top)
        lda = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_top)
        results.append((num_top, [x[1] for x in lda.print_topics(num_topics=20, num_words=20)]))
    return results
