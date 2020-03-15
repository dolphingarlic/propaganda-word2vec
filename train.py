from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import logging
import nltk.data
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


def text_to_wordlist(text, remove_stopwords=False):
    """
    Converts an article into a sequence of words
    """

    # 1. Remove HTML
    # text = BeautifulSoup(text).get_text()

    # 2. Remove non-letters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # 3. Convert words to lowercase and split
    words = text.lower().split()

    # 4. (Optional) Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # 5. Return the list
    return words


# nltk.download()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def text_to_sentences(tweet, tokenizer, remove_stopwords=False):
    """
    Splits an article into parsed sentences
    Returns a list of sentences
    """

    # 1. Use the NLTK tokenizer to split the paragraph
    raw_sentences = tokenizer.tokenize(tweet.strip())

    # 2. Loop over every sentence
    sentences = []
    for sentence in raw_sentences:
        if len(sentence) != 0:
            sentences += text_to_wordlist(sentence, remove_stopwords)

    # 3. Return the list
    return sentences


tweet_file = open('tweets-large.txt', 'r')

clean_tweets = []
for tweet in tweet_file:
    clean_tweets.append(tweet)

sentences = []

for tweet in clean_tweets:
    sentences.append(text_to_sentences(tweet, tokenizer, True))

print(f'There are {len(sentences)} sentences')

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Set values for various parameters
num_features = 1000    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-5   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print('Training model...')
model = Word2Vec(sentences, workers=num_workers, size=num_features,
                 min_count=min_word_count, window=context, sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = 'propaganda-large-2'
model.save(model_name)
