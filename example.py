from collections import Counter
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
import re
import pandas as pd

data_file = pd.read_csv('./Data/material.csv', sep=';', encoding = 'cp1251', error_bad_lines=False,
                        low_memory=False)[['FullName']]

data_file = data_file[['FullName']].astype('str')

morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+/\w+|\w+-\w+|\w+')

rows = [row[0] for row in data_file.values[:10]]
length = {word: len(word) for word in rows}
length = sorted(length.items(), key=lambda item: item[1], reverse=True)

words = []
for row in (rows):
    words.extend([i.lower() for i in tokenizer.tokenize(re.sub(r'\d+', '', row)) if len(i) > 1])
words = [morph.parse(i)[0].normal_form for i in (words) if i == 'то' or i not in stopwords.words('russian')]

word_freq = Counter(words).most_common()
bigrams = Counter(nltk.bigrams(words)).most_common()

bigram_measures = nltk.collocations.BigramAssocMeasures()
f_2 = nltk.collocations.BigramCollocationFinder.from_words(words)
bigrams_pmi = f_2.score_ngrams(bigram_measures.pmi)

df1 = pd.DataFrame({
    'elem': [i for (i, _) in length],
    'len': [i for (_, i) in length]
})

df2 = pd.DataFrame({
    'word': [i for (i, _) in word_freq],
    'freq': [i for (_, i) in word_freq]
})

df3 = pd.DataFrame({
    'bigram': [i for (i, _) in bigrams],
    'freq': [i for (_, i) in bigrams]
})

df4 = pd.DataFrame({
    'bigram_pmi': [i for (i, _) in bigrams_pmi],
    'freq': [i for (_, i) in bigrams_pmi]
})


writer = pd.ExcelWriter(r'./output_nomen.xlsx')
df1.to_excel(writer, 'Length')
df2.to_excel(writer, 'Freq of words')
df3.to_excel(writer, 'Freq of bigrams')
df4.to_excel(writer, 'Bigram collocation')
writer.save()