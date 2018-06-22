
# coding: utf-8

# # 3 Processing Raw Text

# In[1]:


import nltk, re, pprint
from nltk import word_tokenize


# ## 3.1 Accessing text from the web and from disk

# ### Electronic Books

# In[2]:


from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)


# In[3]:


print("Length : {0}".format(len(raw)))
print("raw[:75]: {0}".format(raw[:75]))


# ### Tokenization

# In[4]:


tokens = word_tokenize(raw)
print(type(tokens))
print(len(tokens))
print(tokens[:10])


# In[5]:


text = nltk.Text(tokens)
print(type(text))
print(text[1024:1062])
print(text.collocations())


# In[6]:


# marking the begining and the end
print(raw.find("PART I"))
print(raw.rfind("End of Project Gutenberg"))
raw1 = raw[5336:1157810]
print(raw1.find("PART I"))


# ### Dealing with HTML

# In[7]:


url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
html[:60]


# In[8]:


from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens


# In[9]:


tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance('gene')


# ### Processing RSS Feeds

# In[10]:


import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']


# In[11]:


len(llog.entries)


# In[12]:


post = llog.entries[2]
post.title


# In[13]:


content = post.content[0].value
content[:70]


# In[14]:


raw = BeautifulSoup(content, 'lxml').get_text()
word_tokenize(raw)


# ### Reading Local Files

# In[15]:


prepareFile = open('document.txt', 'w')
prepareFile.write('Time files like an arrow.\nFruit files like a banana.\n')
prepareFile.close()
f = open('document.txt')
f.read()


# In[16]:


f = open('document.txt', 'r')
for line in f:
    print(line.strip())


# In[17]:


# Finding nltk data
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'r')
raw.read()


# ### Capturing User Input

# In[18]:


s = input("Enter some text")


# In[19]:


print("You typed {0} words.".format(len(word_tokenize(s))))


# ### The NLP Pipeline

# In[20]:


raw = open('document.txt').read()
type(raw)


# In[21]:


tokens = word_tokenize(raw)
type(tokens)


# In[22]:


words = [w.lower() for w in tokens]
type(words)


# In[23]:


vocab = sorted(set(words))
type(vocab)


# In[24]:


# We can concatenate strings with strings, and lists with lists, but we cannot concatenate strings with lists:
query = 'Who knows?'
beatles = ['john', 'paul', 'george', 'ringo']
# query + beatles


# ## 3.2 Strings: Text Processing at the Lowest Level

# ### Basic Operations with Strings

# In[25]:


monty = 'Monty Python'
monty


# In[26]:


circus = "Monty Python's Flying Circus"
circus


# In[27]:


circus = 'Monty Python\'s Flying Circus'
circus


# In[28]:


couplet = "Shall I compare then to a Summer's day?""Thou are more lovely and more temperature."
print(couplet)


# In[29]:


couplet = ("Rough winds do shake the darling buds of May,"
          "And Summer's lease hath all too short a date.")
print(couplet)


# In[30]:


couplet = """Shall I compare thee to a Summer's day?
Thou are more lovely and more temprate."""
print(couplet)


# In[31]:


couplet = '''Rough winds do shake the darling buds of May,
And Summer's lease hath all too short a date'''
print(couplet)


# In[32]:


'very' + 'very' + 'very'


# In[33]:


'very' * 3


# In[34]:


a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b: 
    print(line)


# ### Printing Strings

# In[35]:


print(monty)


# In[36]:


grail = 'Holy Grail'
print(monty + grail)


# In[37]:


print(monty, grail)


# In[38]:


print(monty, "and the", grail)


# ### Accessing Individual Characters

# In[39]:


monty[0]


# In[40]:


monty[-1]


# In[41]:


sent = 'colorless green ideas sleep furiously'
for char in sent: 
    print(char, end=' ')


# In[42]:


from nltk.corpus import gutenberg
raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
fdist.most_common(5)


# In[43]:


[char for (char, count) in fdist.most_common()]


# ### Accessing Substrings

# In[44]:


monty[6:10]


# In[45]:


monty[-12:-7]


# In[46]:


monty[:5]


# In[47]:


monty[6:]


# In[48]:


phrase = 'And now for something completely different'
if 'thing' in phrase:
    print('found "thing"')


# In[49]:


monty.find('Python')


# ### The Difference between Lists and Strings

# In[50]:


query = 'Who knows?'
beatles = ['John', 'Paul', 'George', 'Ringo']
query[2]


# In[51]:


beatles[2]


# In[52]:


query[:2]


# In[53]:


beatles[:2]


# In[54]:


query + " I don't"


# In[55]:


beatles + ['Brian']


# In[56]:


beatles[0] = "John Lennon"
del beatles[-1]
beatles


# ## 3.3 Text Processing with Unicode

# ### Extracting encoded text from files

# In[57]:


path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding='latin2')
for line in f: 
    print(line.strip())


# In[58]:


for line in f: 
    line = line.strip()
    print(line.encode('unicode_escape'))


# In[59]:


ord('ń')


# In[60]:


nacute = '\u0144'
nacute


# In[61]:


nacute.encode('utf8')


# In[62]:


import unicodedata
lines = open(path, encoding='latin2').readlines()
line=lines[2]
print(line.encode('unicode_escape'))


# In[63]:


for c in line:
    if ord(c) > 127:
        print('{} U+{:04x} {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c)))


# In[65]:


# Using the re method
line.find('zosta\u0142y')


# In[66]:


line = line.lower()
line


# In[67]:


line.encode('unicode_escape')


# In[68]:


import re
m = re.search('\u015b\w*', line)
m.group()


# In[69]:


# NLTK tokenizers allow Unicode strings as input, and correspondingly yield Unicode strings as output.
word_tokenize(line)


# ## 3.4 Regular Expression for Detecting Word Patterns

# In[72]:


import re 
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]


# ### Using Basic Meta-Characters

# In[73]:


[w for w in wordlist if re.search('ed$', w)]


# In[74]:


[w for w in wordlist if re.search('^..j..t..$', w)]


# ### Ranges and Closures

# In[76]:


# Textonyms
[w for w in wordlist if re.search('^[ghi][mno][jkl][def]$', w)]


# In[77]:


chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]


# In[78]:


wsj = sorted(set(nltk.corpus.treebank.words()))
[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]


# In[79]:


[w for w in wsj if re.search('^[A-Z]+\$$', w)]


# In[82]:


[w for w in wsj if re.search('^[0-9]{4}$', w)]


# In[84]:


[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]


# In[85]:


[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]


# In[86]:


[w for w in wsj if re.search('(ed|ing)$', w)]


# ## 3.5   Useful Applications of Regular Expressions

# ### Extracting Word Pieces

# In[87]:


word = 'supercalifragilisticexpialidocious'
re.findall(r'[aeiou]', word)


# In[88]:


len(re.findall(r'[aeiou]', word))


# In[91]:


wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in wsj
                      for vs in re.findall(r'[aeiou]{2,}', word))
fd.most_common(12)


# ### Doing More with Word Pieces

# In[95]:


regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)

english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))


# In[96]:


rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()


# In[97]:


cv_word_pairs = [(cv, w) for w in rotokas_words
                          for cv in re.findall(r'[ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
cv_index['su']


# In[98]:


cv_index['po']


# ### Finding Word Stems

# In[101]:


def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
        return word
    
re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')


# In[102]:


re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')


# In[103]:


re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')


# In[104]:


re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')


# In[105]:


re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')


# In[106]:


re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language')


# In[107]:


def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)
[stem(t) for t in tokens]


# ### Searching Tokenzied Text

# In[110]:


from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a> (<.*>) <man>")


# In[111]:


chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*> <.*> <bro>")


# In[112]:


chat.findall(r"<l.*>{3,}")


# In[113]:


from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")


# ## 3.6 Normalizing Text

# In[114]:


raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)


# ### Stemmers

# In[115]:


porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]


# In[116]:


[lancaster.stem(t) for t in tokens]


# In[117]:


class IndexedText(object):
    
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                for (i, word) in enumerate(text))
    
    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()
    
porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')


# ### Lemmatization

# In[118]:


wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]


# ## 3.7 Regular Expressions for Tokenizing Text

# ### Simple Approaches to Tokenization

# In[119]:


re.split(r' ', raw)


# In[120]:


re.split(r'[ \t\n]+', raw)


# In[121]:


re.split(r'\W+', raw)


# In[122]:


re.findall(r'\w+|\S\w*', raw)


# In[123]:


print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))


# ### NLTK's Regular Expression Tokenizer

# In[125]:


text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)    # set flag to allow verbose regexps
     ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(-\w+)*        # words with optional internal hyphens
   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
   | \.\.\.            # ellipsis
   | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''
nltk.regexp_tokenize(text, pattern)


# ## 3.8   Segmentation

# ### Sentence Segmentation

# In[126]:


len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())


# In[127]:


text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
pprint.pprint(sents[79:89])


# ## Word Segmentation

# In[129]:


text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"


# In[130]:


def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
segment(text, seg1)


# In[131]:


segment(text, seg2)


# In[132]:


def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(word) + 1 for word in set(words))
    return text_size + lexicon_size

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"
segment(text, seg3)


# In[133]:


evaluate(text, seg3)


# In[134]:


from random import randint

def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
anneal(text, seg1, 5000, 1.2)


# ## 3.9   Formatting: From Lists to Strings

# ### From Lists to Strings

# In[135]:


silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
' '.join(silly)


# In[136]:


';'.join(silly)


# In[137]:


''.join(silly)


# ### Strings and Formats

# In[138]:


word = 'cat'
sentence = """hello
 world"""
print(word)


# In[139]:


word


# In[140]:


sentence


# In[141]:


fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
for word in sorted(fdist):
    print(word, '->', fdist[word], end='; ')


# In[142]:


for word in sorted(fdist):
    print('{}->{};'.format(word, fdist[word]), end=' ')


# In[143]:


'{}->{};'.format ('cat', 3)


# In[144]:


'{}->'.format('cat')


# In[145]:


'{}'.format(3)


# In[146]:


'I want a {} right now'.format('coffee')


# In[147]:


'{} wants a {} {}'.format ('Lee', 'sandwich', 'for lunch')


# In[148]:


'{} wants a {}'.format ('Lee', 'sandwich', 'for lunch')


# In[149]:


'from {1} to {0}'.format('A', 'B')


# In[150]:


template = 'Lee wants a {} right now'
menu = ['sandwich', 'spam fritter', 'pancake']
for snack in menu:
     print(template.format(snack))


# ### Lining Things Up

# In[151]:


'{:6}'.format(41)


# In[152]:


'{:<6}' .format(41)


# In[153]:


'{:6}'.format('dog')


# In[154]:


'{:>6}'.format('dog')


# In[156]:


import math
'{:.4f}'.format(math.pi)


# In[157]:


count, total = 3205, 9375
"accuracy for {} words: {:.4%}".format(total, count / total)


# In[158]:


def tabulate(cfdist, words, categories):
    print('{:16}'.format('Category'), end=' ')                    # column headings
    for word in words:
        print('{:>6}'.format(word), end=' ')
    print()
    for category in categories:
        print('{:16}'.format(category), end=' ')                  # row heading
        for word in words:                                        # for each word
            print('{:6}'.format(cfdist[category][word]), end=' ') # print table cell
        print()                                                   # end the row

from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)


# ### Writing Results to a File

# In[160]:


output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
     print(word, file=output_file)


# In[161]:


len(words)
str(len(words))
print(str(len(words)), file=output_file)


# ### Text Wrapping

# In[162]:


saying = ['After', 'all', 'is', 'said', 'and', 'done', ',',
           'more', 'is', 'said', 'than', 'done', '.']
for word in saying:
     print(word, '(' + str(len(word)) + '),', end=' ')


# In[163]:


from textwrap import fill
format = '%s (%d),'
pieces = [format % (word, len(word)) for word in saying]
output = ' '.join(pieces)
wrapped = fill(output)
print(wrapped)

