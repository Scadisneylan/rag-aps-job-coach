import nltk
from nltk.tokenize import word_tokenize

# Your text
text = "Hello there, this is a sample sentence for tokenization using NLTK. It's quite powerful!"

# Tokenize the text into words
tokens = word_tokenize(text)

print("Original Text:", text)
print("Word Tokens:", tokens)