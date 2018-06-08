
# coding: utf-8

# In[5]:


# 1.2   Getting Started with NLTK

from nltk.book import *

text1


# In[9]:


# 1.3   Searching Text

text1.concordance("monstrous")


# In[8]:


text1.similar("monstrous")


# In[7]:


text2.similar("monstrous")


# In[15]:


text2.common_contexts(["monstrous", "very"])


# In[18]:


get_ipython().run_line_magic('matplotlib', 'notebook')

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# In[23]:


# 1.4   Counting Vocabulary

# Total number of words
len(text3)


# In[24]:


# Sorted list of unique words
sorted(set(text3))


# In[27]:


# Total number of unique words
len(set(text3))


# In[28]:


# Measure of lexical richness of text
len(set(text3)) / len(text3)


# In[29]:


text3.count("smote")


# In[31]:


# Percentage of the text taken up by the specific word.
100 * text4.count("a") / len(text4)


# In[32]:


# functions:
def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total): 
    return 100 * count / total


# In[40]:


print("Lexical diversity for text3: " + str(lexical_diversity(text3)))
print("Percentage of 'a' for text: " + str(percentage(text4.count('a'), len(text4))))


# In[43]:


# 2.2   Indexing Lists

text4[173]


# In[44]:


text4.index('awaken')


# In[46]:


# Slicing

text5[16715:16735]


# In[50]:


# 3.1 Frequency Distribution

fdist1 = FreqDist(text1)
print(fdist1)


# In[51]:


fdist1.most_common(50)


# In[52]:


# Frequency of word whale 

fdist1['whale']


# In[60]:


get_ipython().run_line_magic('matplotlib', 'notebook')

fdist1.plot(50, cumulative=True)


# In[8]:


# 3.2 Fine-grained Selection of Words

# Words from the vocabulary of the text that are more than 15 characters long.

V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# In[11]:


""" Frequently occuring long words 
    All Words from chat corpus that are longer than 7 characters, that occur more than 7 times
"""

fdists = FreqDist(text5)
sorted(w for w in set(text5) if len(w) > 7 and fdists[w] > 7)


# In[12]:


# 3.3 Collocations and Bigrams

# extract list of bigrams
list(bigrams(['more', 'is', 'said', 'than', 'done']))


# In[18]:


# bigrams that occur more often than we would expect based on the frequency of the individual words - collocations
print(text4.collocations())
print(text5.collocations())


# In[20]:


# 3.4 Counting Other Things
 
# The distribution of word lengths in a text
print([len(w) for w in text1])
fdist = FreqDist(len(w) for w in text1)
print(fdist)


# In[21]:


fdist


# In[22]:


fdist.most_common()


# In[23]:


fdist.max()


# In[24]:


fdist[3]


# In[25]:


fdist.freq(3)


# In[26]:


# 4 Back to PYthon: Making Decisions and Taking Control
# 4.1 Conditionals

# We can use different operators to select different words from a sentence of news text
sent7


# In[27]:


[w for w in sent7 if len(w) < 4]


# In[28]:


[w for w in sent7 if len(w) <= 4]


# In[29]:


[w for w in sent7 if len(w) == 4]


# In[30]:


[w for w in sent7 if len(w) != 4]


# In[41]:


# Word Comparison Operators
print('1) Result of endswith -> {0} \n'.format(sorted(w for w in set(text1) if w.endswith('ableness'))))
print('2) Result of in -> {0} \n'.format(sorted(term for term in set(text4) if 'gnt' in term)))
print('3) Result of istitle -> {0} \n'.format(sorted(item for item in set(text6) if item.istitle())))
print('4) Result of isdigit -> {0} \n'.format(sorted(item for item in set(sent7) if item.isdigit())))


# In[44]:


# More complex conditions
print('1) -> {0} \n'.format(sorted(w for w in set(text7) if '-' in w and 'index' in w)))
print('2) -> {0} \n'.format(sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10)))
print('3) -> {0} \n'.format(sorted(w for w in set(text7) if not w.islower())))
print('4) -> {0} \n'.format(sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)))


# In[46]:


# 4.2 Operating on Every Element
print('1) -> {0} \n'.format([len(w) for w in text1]))
print('1) -> {0} \n'.format([w.upper() for w in text1]))


# In[49]:


# Vocabulary size
print(len(text1))
print(len(set(text1)))
print(len(set(word.lower() for word in text1)))
print(len(set(word.lower() for word in text1 if word.isalpha())))


# In[72]:


# 4.3 Nested Code Blocks
word = 'cat'
if len(word) > 5:
    print('word length is less than 5')
    
for word in ['Call', 'me', 'Ishmael']:
    print(word)


# In[75]:


# 4.4 Looping with Conditions
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)


# In[77]:


for token in sent1:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')


# In[78]:


# Combine the idioms
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')


# In[ ]:




