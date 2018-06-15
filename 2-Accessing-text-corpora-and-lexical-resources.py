
# coding: utf-8

# In[1]:


# 1. Accessing Text Corpora
from nltk.book import *


# In[2]:


# 1.1 Gutenberg Corpus
import nltk

# the file identifiers in this corpus
nltk.corpus.gutenberg.fileids()


# In[3]:


# Picking out texts and counting the words
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)


# In[4]:


# using concordance for text other than in nltk.book import *
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")


# In[5]:


# alternative short from to import nltk corpus
from nltk.corpus import gutenberg
print("File identifiers: {0}\n".format(gutenberg.fileids()))
print("emma: {0}\n".format(gutenberg.words('austen-emma.txt')))


# In[6]:


# Displaying other information about each text in gutenberg file
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    # Displays average word length, average sentence length, and the number of times each vocab item appears in the text on average
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)


# In[7]:


# extracting sentences
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
print("All Sentences {0}\n".format(macbeth_sentences))
print("Sentence number 1116 {0}\n".format(macbeth_sentences[1116]))

longest_len = max(len(s) for s in macbeth_sentences)
[s for s in macbeth_sentences if len(s) == longest_len]


# In[8]:


# 1.2 Web and Chat Text
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')


# In[9]:


# chat session corpus baseeed on date chatroom and number of posts
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]


# In[10]:


# 1.3 Brown Corpus
from nltk.corpus import brown
print("Categories {0} \n".format(brown.categories()))
print("List of words by category {0} \n".format(brown.words(categories='news')))
print("List of words by fileid {0} \n".format(brown.words(fileids=['cg22'])))
print("List of sentences by category {0} \n".format(brown.sents(categories=['news', 'editorial', 'reviews'])))


# In[11]:


# using brown corpus for studying systematic differences between genres
# first step : produce counts for a particular genre
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals: 
    print(m + ':', fdist[m], end=' ')


# In[12]:


# obtain counts for each genre of interest using cfd
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)


# In[13]:


# 1.4 Reuters Corpus
from nltk.corpus import reuters
print(reuters.fileids())
print(reuters.categories())


# In[14]:


print("Accessing through single category {0} \n".format(reuters.categories('training/9865')))
print("Accessing through multiple category {0} \n".format(reuters.categories(['training/9865', 'training/9880'])))
print("Accessing through single fileid {0} \n".format(reuters.fileids('barley')))
print("Accessing through multiple fileids {0} \n".format(reuters.fileids(['barley', 'corn'])))


# In[15]:


# Words or sentences in terms of files or categories
print("Accessing words {0} \n".format(reuters.words('training/9865')[:14]))
print("Accessing words {0} \n".format(reuters.words(['training/9865', 'training/9880'])))
print("Accessing words {0} \n".format(reuters.words(categories='barley')))
print("Accessing words {0} \n".format(reuters.words(categories=['barley', 'corn'])))


# In[16]:


# 1.5 Inaugural Address Corpus
from nltk.corpus import inaugural
print(inaugural.fileids())
[fileid[:4] for fileid in inaugural.fileids()]


# In[17]:


# How words americal and citizen are used over time
get_ipython().run_line_magic('matplotlib', 'notebook')
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd.plot()


# In[18]:


# 1.7 Corpora in Other Languages
print(nltk.corpus.cess_esp.words())
print(nltk.corpus.floresta.words())
print(nltk.corpus.indian.words('hindi.pos'))
print(nltk.corpus.udhr.fileids())
print(nltk.corpus.udhr.words('Javanese-Latin1')[11:])


# In[19]:


# differences in word lengths for a selection of languages included in udhr
get_ipython().run_line_magic('matplotlib', 'notebook')
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)


# In[23]:


# 1.8 Text Corpus structure
raw = gutenberg.raw("burgess-busterbrown.txt")
raw[1:20]


# In[24]:


words = gutenberg.words("burgess-busterbrown.txt")
words[1:20]


# In[25]:


sents = gutenberg.sents("burgess-busterbrown.txt")
sents[1:20]


# In[26]:


# 1.9 Loading your own corpus
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()


# In[28]:


wordlists.words('american-english')


# In[33]:


# 2 Conditional Frequency Distributions
# 2.2 Counting words by genre
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
cfd


# In[35]:


# Looking at only two genres
genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
len(genre_word)


# In[36]:


# Begining part will contain news category
genre_word[:4]


# In[37]:


# End part will contain romance category
genre_word[-4:]


# In[40]:


# We can now user this list of pairs to create a conditionalFreqdist
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd)
cfd.conditions()


# In[43]:


# accessing the two categories
print("CFD of news {0} ".format(cfd['news']))
print("CFD of romance {0} ".format(cfd['romance']))
print("CFD of most_commont 20 in romance {0} ".format(cfd['romance'].most_common(20)))
print("number of 'could' in romance {0} ".format(cfd['romance']['could']))


# In[47]:


# 2.3 Plotting and Tabulating distributions
from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd


# In[51]:


from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'],
                samples=range(10), cumulative=True)


# In[54]:


# 2.4 Generating Random Text with Bigrams
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven','and', 'the', 'earth', '.']
list(nltk.bigrams(sent))


# In[62]:


def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()
        
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
cfd['living']


# In[63]:


generate_model(cfd, 'living')


# In[65]:


# 3 More python: reusing code
# 3.2 Functions
from __future__ import division
def lexical_diversity(text):
    return len(text) / len(set(text))

def lexical_diversity_multiline(text):
    word_count = len(text)
    vocab_size = len(set(text))
    diversity_score = vocab_size / word_count
    return diversity_score

from nltk.corpus import genesis
kjv = genesis.words('english-kjv.txt')
lexical_diversity_multiline(kjv)


# In[72]:


def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

print(plural('fairy'))
print(plural('woman'))
print(plural('wish'))
print(plural('fan'))


# In[76]:


# 4 Lexical Resources
# 4.1 Wordlist Corpora
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))


# In[77]:


unusual_words(nltk.corpus.nps_chat.words())


# In[78]:


# stopwords
from nltk.corpus import stopwords
stopwords.words('english')


# In[81]:


def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())


# In[84]:


# Word puzzle
puzzle_letters = nltk.FreqDist('eqivrvonl')
obligatory = 'r'
wordList = nltk.corpus.words.words()
[w for w in wordList if len(w) >= 6
                     and obligatory in w 
                     and nltk.FreqDist(w) <= puzzle_letters]


# In[85]:


# Names that are ambiguous for gender
names = nltk.corpus.names
names.fileids()


# In[86]:


male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]


# In[87]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1])
    for fileid in names.fileids()
    for name in names.words(fileid))
cfd.plot()


# In[88]:


# 4.2 A Pronouncing Dictionary
entries = nltk.corpus.cmudict.entries()
len(entries)


# In[89]:


for entry in entries[42371:42379]:
    print(entry)


# In[91]:


# lexicon for entries whose pronunciation consists of 3 phones
for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2, end=' ')


# In[92]:


# all words whose pronunciation ends with a syllable sounding like nicks
syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]


# In[93]:


[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']


# In[94]:


sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))


# In[95]:


def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]

[w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]


# In[96]:


[w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]


# In[97]:


# find minimally-contrasting sets of words
p3 = [(pron[0] + '-' + pron[2], word)
     for (word, pron) in entries
     if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
    if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring[:70] + "...")


# In[99]:


prondict = nltk.corpus.cmudict.dict()
prondict['fire']


# In[101]:


prondict['blog'] = [['B', 'L', 'AA1', 'G']]
prondict['blog']


# In[104]:


# process a text, e.g., to filter out words having some lexical property (like nouns), or mapping every word of the text. 
text = ['natural', 'language', 'processing']
[ph for w in text for ph in prondict[w][0]]


# In[106]:


# 4.3 Comparative Wordlists
from nltk.corpus import swadesh
print(swadesh.fileids())
swadesh.words('en')


# In[109]:


# We can access cognate words from multiple languages using the entries() method, specifying a list of languages
fr2en = swadesh.entries(['fr', 'en'])
print(fr2en)
translate = dict(fr2en)
print(translate['chien'])
print(translate['jeter'])


# In[112]:


de2en = swadesh.entries(['de', 'en'])
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(de2en))
translate.update(dict(es2en))
print(translate['Hund'])
print(translate['perro'])


# In[113]:


# We can compare words in various Germanic and Romance languages
languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])


# In[114]:


# Shoebox and Toolbox Lexicons
from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')


# In[116]:


# 5 WordNet
# 5.1 Senses and Synonyms
from nltk.corpus import wordnet as wn
# synonym set for motorcar
wn.synsets('motorcar')


# In[117]:


wn.synset('car.n.01').lemma_names()


# In[118]:


wn.synset('car.n.01').definition()


# In[119]:


wn.synset('car.n.01').examples()


# In[120]:


wn.synset('car.n.01').lemmas()


# In[123]:


print(wn.lemma('car.n.01.automobile'))
print(wn.lemma('car.n.01.automobile').synset())
print(wn.lemma('car.n.01.automobile').name())


# In[124]:


# the word car is ambiguous, having five synsets:
wn.synsets('car')


# In[125]:


for synset in wn.synsets('car'):
    print(synset.lemma_names())


# In[126]:


# we can access all the lemmas involving the word car as follows.
wn.lemmas('car')


# In[127]:


# 5.2 The WordNet Hierarchy
# WordNet makes it easy to navigate between concepts.
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0]


# In[129]:


sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())


# In[131]:


# We can also navigate up the hierarchy by visiting hypernyms
motorcar.hypernyms()
paths = motorcar.hypernym_paths()
len(paths)


# In[132]:


[synset.name() for synset in paths[0]]


# In[133]:


[synset.name() for synset in paths[1]]


# In[134]:


# We can get the most general hypernyms (or root hypernyms) of a synset as follows:
motorcar.root_hypernyms()


# In[135]:


# 5.3 More Lexical Relations
wn.synset('tree.n.01').part_meronyms()


# In[136]:


wn.synset('tree.n.01').substance_meronyms()


# In[137]:


wn.synset('tree.n.01').member_holonyms()


# In[139]:


# the word mint, which has several closely-related senses.
for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())


# In[140]:


wn.synset('mint.n.04').part_holonyms()


# In[143]:


wn.synset('mint.n.04').substance_holonyms()


# In[146]:


# There are also relationships between verbs
print(wn.synset('walk.v.01').entailments())
print(wn.synset('eat.v.01').entailments())
print(wn.synset('tease.v.03').entailments())


# In[152]:


# Some lexical relationships hold between lemmas, e.g., antonymy:
print(wn.lemma('supply.n.02.supply').antonyms())
print(wn.lemma('rush.v.01.rush').antonyms())
print(wn.lemma('horizontal.a.01.horizontal').antonyms())
print(wn.lemma('staccato.r.01.staccato').antonyms())


# In[156]:


# 5.4 Semantic Similarity
# If two synsets share a very specific hypernym — one that is low down in the hypernym hierarchy — they must be closely related.
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
print(right.lowest_common_hypernyms(minke))
print(right.lowest_common_hypernyms(orca))
print(right.lowest_common_hypernyms(tortoise))
print(right.lowest_common_hypernyms(novel))


# In[157]:


# We can quantify the concept of generality by looking up the depth of each synset:
print(wn.synset('baleen_whale.n.01').min_depth())
print(wn.synset('whale.n.02').min_depth())
print(wn.synset('vertebrate.n.01').min_depth())
print(wn.synset('entity.n.01').min_depth())


# In[158]:


# the number decrease as we move away from the semantic space of sea creatures to inanimate objects
print(right.path_similarity(minke))
print(right.path_similarity(orca))
print(right.path_similarity(tortoise))
print(right.path_similarity(novel))

