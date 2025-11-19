#  Data Pre-Processing Techniques in NLP
# 
#
# 1) Tokenization : ---> Breaking the words into small chunks 
#   
#   small chunks are called tokens 
# 
#     Sentence tokenization
#     Word Tokenization
#     Sub-word tokenization


#      Tokenization techniques 
#            1. Rule-Based Techniques
#            2. Statistical Tokenization
#            


# Word Tokenization

from nltk.tokenize import word_tokenize

text="Tokenization is a key process in NLP. It breaks text into words and sentences"
print("Text : ", text)
word_tokens=word_tokenize(text)
print("Word Tokens :", word_tokens)



# Sentence Tokenization

from nltk.tokenize import sent_tokenize

sentence_tokens = sent_tokenize(text)
print("Sentence Tokens : ", sentence_tokens)


# sub-word tokenization
word="tokenization"
subword_tokens=[word[i:i+2] for i in range(0,len(word),2)]
print(f"\nExample: \"{word}\" -> {subword_tokens}")





# Rule-Based Tokenization

"""for Rule-Based tokenization, we can create custom rules using regular expressions."""

import re

rule_text="Tokenization: splitting text into words, phrases, or other meaningful elements."
print("Rule Text : ",rule_text)
rule_tokens=re.findall(r'\b\w+\b', rule_text)
print("Rule-Based Tokens : ", rule_tokens)





# Statistical Based Tokenization

"""Statistical Tokenization, commanly reffere to as Sentence Boundary Detection (580),
primarily involves using statistical models to determine sentence boundaries in a text."""

from nltk.tokenize import PunktSentenceTokenizer
print("Statistical Text : ", text)
punkt_tokenizer= PunktSentenceTokenizer()
statistical_tokens=punkt_tokenizer.tokenize(text)
print("Statistical Tokens : ", statistical_tokens)
