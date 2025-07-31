import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet # Needed for WordNetLemmatizer's POS mapping

# Make sure you've downloaded these:
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet') # Already did this, but good to ensure

text = "Hello there, this is a sample sentence for tokenization using NLTK. It's quite powerful!"

# 1. Word Tokenization
tokens = word_tokenize(text)
print("1. Original Word Tokens:", tokens)

# 2. Lowercasing
lower_tokens = [word.lower() for word in tokens]
print("2. Lowercased Tokens:", lower_tokens)

# 3. Removing Punctuation
no_punctuation_tokens = [word for word in lower_tokens if word not in string.punctuation]
print("3. Tokens without Punctuation:", no_punctuation_tokens)

# 4. Removing Stop Words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in no_punctuation_tokens if word not in stop_words]
print("4. Filtered Tokens (Stop words removed):", filtered_tokens)

# 5. Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("5. Stemmed Tokens:", stemmed_tokens)

# 6. Lemmatization (more accurate than stemming)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN # Default

lemmatized_tokens = []
pos_tagged_tokens = pos_tag(filtered_tokens) # Need POS tags for good lemmatization

for word, tag in pos_tagged_tokens:
    w_pos = get_wordnet_pos(tag)
    lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=w_pos))
print("6. Lemmatized Tokens:", lemmatized_tokens)