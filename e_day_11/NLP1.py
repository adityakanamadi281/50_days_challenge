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




# Named_Entity_Recognition : ----> It is NLP technique that aims to identify and classify entities, such as names of people, organizations.


# NER helps to extract key information from text, enhancing the understanding of documents. Facilitates efficient searching and retrieval of relavant information.
# Essential for systens that answers questions by extracting information from texts.


#  How NER works:
#   - Tokenization: Breaks the text into individual words or tokens.
#   - Part-of-Speech tagging : Assign a part-of-speech tag to each token.
#   - NER : Identify and classify tokens into predefined categories(Entities).
# 


#    Types of Entities:
#       1. Person
#       2. Organization
#       3. Location
#       4. Date 
#       5. Time
#       6. Money
#       7. Percentage



import spacy


nlp = spacy.load("en_core_web_sm")

text1= "Sundar Pichai is the CEO of Google and lives in California."

doc= nlp(text1)

entities = [(ent.text, ent.label_) for ent in doc.ents]

print("Named Entities : ")
for entity, label in entities:
    print(f"{label}: {entity}")
    


# Stopword Removal : It is text preprocessing step where common words that do not contribute significant meaning to the text are eliminated.


#  Working:
#    1. Tokenization
#    2. Stopwords
#    3. Removal


import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


text2 = "Stopword removal is an important step in natural language processing. It helps improve text analysis."

words = word_tokenize(text2)

stop_words = set(stopwords.words("english"))

filtered_words = [word for word in words if word.lower() not in stop_words]

print("original Words : ", words)
print("Filtered words (without stopwords): ", filtered_words)