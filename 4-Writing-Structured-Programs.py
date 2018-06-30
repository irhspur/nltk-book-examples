
# coding: utf-8

# In[1]:


import nltk, re, pprint
from nltk import word_tokenize


# ## 4.1 Back to the Basics

# ### Assignment

# In[2]:


foo = 'Monty'
bar = foo
foo = 'Python'
bar


# In[3]:


foo = ['Monty', 'Python']
bar = foo
foo[1] = 'Bodkin'
bar


# In[4]:


empty = []
nested = [empty, empty, empty]
nested


# In[5]:


nested[1].append('Python')
nested


# In[6]:


nested = [[]] * 3
nested[1].append('Python')
nested[1] = ['Monty']
nested


# ### Equality

# In[7]:


size = 5
python = ['Python']
snake_nest = [python] * size
snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]


# In[8]:


snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]


# In[9]:


import random
position = random.choice(range(size))
snake_nest[position] = ['Python']
snake_nest


# In[10]:


snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]


# In[11]:


snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]


# In[12]:


[id(snake) for snake in snake_nest]


# ### Conditionals

# In[13]:


mixed = ['cat', '', ['dog'], []]
for element in mixed:
    if element:
        print(element)


# In[14]:


animals = ['cat', 'dog']
if 'cat' in animals: 
    print(1)
elif 'dog' in animals:
    print(2)
    


# In[15]:


sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
all(len(w) > 4 for w in sent)


# In[16]:


any(len(w) > 4 for w in sent)


# ## 4.2 Sequences

# In[17]:


t = 'walk', 'fem', 3
t


# In[18]:


t[0]


# In[19]:


t[1:]


# In[20]:


len(t)


# In[21]:


raw = 'I turned off the spectroroute'
text = ['I', 'turned', 'off', 'the', 'spectroroute']
pair = (6, 'turned')
raw[2], text[3], pair[1]


# In[22]:


raw[-3:], text[-3:], pair[-3:]


# In[23]:


len(raw), len(text), len(pair)


# ### Operating of Sequence Types

# In[24]:


raw = 'Red lorry, yellow lorry, red lorry, yellow lorry'
text = word_tokenize(raw)
fdist = nltk.FreqDist(text)
sorted(fdist)


# In[25]:


for key in fdist: 
    print(key + ':', fdist[key], end='; ')


# In[26]:


words = ['I', 'turned', 'off', 'the', 'spectroroute']
words[2], words[3], words[4] = words[3], words[4], words[2]
words


# In[27]:


tmp = words[2]
words[2] = words[3]
words[3] = words[4]
words[4] = tmp


# In[28]:


words = ['I', 'turned', 'off', 'the', 'spectroroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
zip(words, tags)


# In[29]:


list(zip(words, tags))


# In[30]:


list(enumerate(words))


# In[31]:


text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
text == training_data + test_data


# In[32]:


len(training_data) / len(test_data)


# ### Combining Different Sequence Types

# In[33]:


words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
wordlens.sort()
' '.join(w for (_, w) in wordlens)


# In[34]:


lexicon = [
    ('the', 'det', ['Di:', 'D@']),
    ('off', 'prep', ['Qf', 'O:f'])
]
lexicon.sort()
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
del lexicon[0]


# ### Generator Expressions

# In[35]:


text = '''"When I use a word," Humpty Dumpty said in rather scornful 
tone. "it means just what I choose it to mean - neither more nor less"'''
[w.lower() for w in word_tokenize(text)]


# In[36]:


max([w.lower() for w in word_tokenize(text)])


# In[37]:


max(w.lower() for w in word_tokenize(text))


# ## 4.3 Questions of Style

# ### Procedural vs Declarative Style

# In[38]:


tokens = nltk.corpus.brown.words(categories='news')
count = 0
total = 0
for token in tokens:
    count += 1
    total += len(token)
total / count
                                 


# In[41]:


total = sum(len(t) for t in tokens)
print(total / len(tokens))
print(len(tokens))


# In[40]:


word_list = []
i = 0
while i < len(tokens):
    j = 0
    while j < len(word_list) and word_list[j] <= tokens[i]:
        j += 1
    if j == 0 or tokens[i] != word_list[j-1]:
        word_list.insert(j, tokens[i])
    i += 1


# In[42]:


word_list = sorted(set(tokens))


# In[43]:


fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
        break


# In[44]:


text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
     if len(word) > len(longest):
         longest = word
longest


# In[45]:


maxlen = max(len(word) for word in text)
[word for word in text if len(word) == maxlen]


# ### Some Legitimate Uses for Counters

# In[46]:


sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)]


# In[47]:


m, n = 3, 7
array = [[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
pprint.pprint(array)


# In[48]:


array = [[set()] * n ] * m
array[2][5].add(7)
pprint.pprint(array)


# ## 4.4 Functions: The Foundation of Structured Programming

# In[49]:


import re
def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file).read()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

help(get_text)


# ### Function Inputs and Output

# In[50]:


def repeat(msg, num):
    return ' '.join([msg] * num)
monty = 'Monty Python'
repeat(monty, 3)


# In[51]:


def monty():
    return 'Monty Python'
monty()


# In[52]:


repeat(monty(), 3)


# In[53]:


def my_sort1(mylist): # good: modifies its argument, no return value
    mylist.sort()
    
def my_sort2(mylist): # good: doesn't touch its argumet, returns value
    return sorted(mylist)

def my_sort3(mylist): #bad: modifies its argument and also returns it
    mylist.sort()
    return mylist


# ### Parameter Passing

# In[54]:


def set_up(word, properties):
    word = 'lolcat'
    properties.append('noun')
    properties = 5
    
w = ''
p = []
set_up(w, p)
w


# In[55]:


p


# ### Checking Parameter Types

# In[56]:


def tag(word):
    if word in ['a', 'the', 'all']:
        return 'det'
    else: 
        return 'noun'
# Here function assumed that its argument would always be a string
tag('the')


# In[57]:


tag('knight')


# In[58]:


tag(["'Tis", 'but', 'a', 'scratch'])


# In[59]:


def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else: 
        return 'noun'
    


# ### Functional Decomposition

# In[60]:


from urllib import request
from bs4 import BeautifulSoup

def freq_words(url, freqdist, n):
    html = request.urlopen(url).read().decode('utf-8')
    raw = BeautifulSoup(html, "lxml").get_text()
    for word in word_tokenize(raw):
        freqdist[word.lower()] += 1
    result = []
    for word, count in freqdist.most_common(n):
        result = result + [word]
    print(result)

constitution = "http://www.archives.gov/exhibits/charters/constitution_transcript.html"
fd = nltk.FreqDist()
freq_words(constitution, fd, 30)


# In[61]:


from urllib import request
from bs4 import BeautifulSoup

def freq_words(url, n):
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html, 'lxml').get_text()
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(text))
    return [word for (word, _) in freqdist.most_common(n)]

freq_words(constitution, 30)


# ### Documenting Functions

# In[62]:


def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.

    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.

        >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
        0.5

    :param reference: An ordered list of reference values
    :type reference: list
    :param test: A list of values to compare against the corresponding
        reference values
    :type test: list
    :return: the accuracy score
    :rtype: float
    :raises ValueError: If reference and length do not have the same length
    """

    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in zip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)


# ## 4.5 Doing More with Functions

# ### Functions as Arguments

# In[63]:


sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the', 'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
def extract_property(prop):
    return [prop(word) for word in sent]

extract_property(len)


# In[64]:


def last_letter(word):
    return word[-1]
extract_property(last_letter)


# #### Lamda Expressions

# In[65]:


extract_property(lambda w: w[-1])


# In[66]:


sorted(sent)


# In[67]:


sorted(sent, key=lambda x: len(x), reverse=True)


# ### Accumulative Functions

# In[68]:


def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result

def search2(substring, words):
    for word in words:
        if substring in word:
            yield word

for item in search1('zz', nltk.corpus.brown.words()):
    print(item, end = " ")


# In[69]:


for item in search2('zz', nltk.corpus.brown.words()):
    print(item, end = " ")


# In[70]:


def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else: 
        for perm in permutations(seq[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]

list(permutations(['police', 'fish', 'buffalo']))


# ### Higher-Order Functions

# In[71]:


def is_content_word(word):
     return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
         'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
list(filter(is_content_word, sent))


# In[72]:


[w for w in sent if is_content_word(w)]


# In[73]:


lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))
sum(lengths) / len(lengths)


# In[74]:


lengths = [len(sent) for sent in nltk.corpus.brown.sents(categories='news')]
sum(lengths) / len(lengths)


# In[75]:


# list(map(lambda w: len(filter(lambda c: c.lower() in "aeiou", w)), sent))
# [len(c for c in w if c.lower() in "aeiou") for w in sent]


# In[76]:


def repeat(msg='<empty>', num=1):
     return msg * num
repeat(num=3)


# In[77]:


repeat(msg='Alice')


# In[78]:


repeat(num=5, msg='Alice')


# In[79]:


def generic(*args, **kwargs):
    print(args)
    print(kwargs)
generic(1, "African swallow", monty="python")


# In[80]:


song = [['four', 'calling', 'birds'],
         ['three', 'French', 'hens'],
         ['two', 'turtle', 'doves']]
list(zip(song[0], song[1], song[2]))


# In[81]:


list(zip(*song))


# In[82]:


def freq_words(file, min=1, num=10):
    text = open(file).read()
    tokens = word_tokenize(text)
    freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
    return freqdist.most_common(num)

f = open('ch01.rst', 'w');
f.write('as the as the sky turns red, the moon will die')
f.close()
fw = freq_words('ch01.rst', 4, 10)
fw = freq_words('ch01.rst', min=4, num=10)
fw = freq_words('ch01.rst', num=10, min=4)


# In[83]:


def freq_words(file, min=1, num=10, verbose=False):
    freqdist = nltk.FreqDist()
    if verbose: print("Opening", file)
    text = open(file).read()
    if verbose: print("Read in %d characters" % len(file))
    for word in word_tokenize(text):
        if len(word) >= min:
            freqdist[word] += 1
            if verbose and freqdist.N() % 100 == 0: print(".", sep="")
    if verbose: print
    return freqdist.most_common(num)

freq_words('ch01.rst', verbose=True)


# ## 4.6 Program Development

# ### Sources of Error

# In[84]:


def find_words(text, wordlength, result=[]):
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result

find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 3)


# In[85]:


find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 2, ['ur'])


# In[86]:


find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 3)


# In[87]:


#import pdb
#pdb.run("find_words(['cat'], 3)")


# ### Recursion

# In[88]:


def factorial1(n):
    result = 1
    for i in range(n):
        result += (i + 1)
    return result


# In[89]:


def factorial2(n):
    if n == 1:
        return 1
    else:
        return n * factorial2(n - 1)


# In[90]:


def size1(s):
    return 1 + sum(size1(child) for child in s.hyponyms())


# In[91]:


def size2(s):
    layer = [s]
    total = 9
    while layer: 
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total


# In[92]:


from nltk.corpus import wordnet as wn
dog = wn.synset('dog.n.01')
size1(dog)


# In[93]:


size2(dog)


# In[94]:


def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value

trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
trie = dict(trie)               # for nicer printing
trie['c']['h']['a']['t']['value']


# In[95]:


pprint.pprint(trie, width=40)


# ### Space-Time Tradeoffs

# In[96]:


def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub('\s+', ' ', contents)
    return contents

def snippet(doc, term):
    text = ' '*30 + raw(doc) + ' '*30
    pos = text.index(term)
    return text[pos-30:pos+30]

print("Building Index...")
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w, f) for f in files for w in raw(f).split())

query = ''
while query != "quit":
    query = input("query> ")     # use raw_input() in Python 2
    if query in idx:
        for doc in idx[query]:
            print(snippet(doc, query))
    else:
        print("Not found")


# In[97]:


def preprocess(tagged_corpus):
    words = set()
    tags = set()
    for sent in tagged_corpus:
        for word, tag in sent:
            words.add(word)
            tags.add(tag)
    wm = dict((w, i) for (i, w) in enumerate(words))
    tm = dict((t, i) for (i, t) in enumerate(tags))
    return [[(wm[w], tm[t]) for (w, t) in sent] for sent in tagged_corpus]


# In[99]:


from timeit import Timer
vocab_size = 100000
setup_list = "import random; vocab = range(%d)" % vocab_size
setup_set = "import random; vocab = set(range(%d))" % vocab_size
statement = "random.randint(0, %d) in vocab" % (vocab_size * 2)
print(Timer(statement, setup_list).timeit(1000))
print('\n')
print(Timer(statement, setup_set).timeit(1000))


# In[100]:


def virahanka1(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka1(n-1)]
        l = ["L" + prosody for prosody in virahanka1(n-2)]
        return s + l

def virahanka2(n):
    lookup = [[""], ["S"]]
    for i in range(n-1):
        s = ["S" + prosody for prosody in lookup[i+1]]
        l = ["L" + prosody for prosody in lookup[i]]
        lookup.append(s + l)
    return lookup[n]

def virahanka3(n, lookup={0:[""], 1:["S"]}):
    if n not in lookup:
        s = ["S" + prosody for prosody in virahanka3(n-1)]
        l = ["L" + prosody for prosody in virahanka3(n-2)]
        lookup[n] = s + l
    return lookup[n]

from nltk import memoize
@memoize
def virahanka4(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n-1)]
        l = ["L" + prosody for prosody in virahanka4(n-2)]
        return s + l


# In[101]:


virahanka1(4)


# In[102]:


virahanka2(4)


# In[103]:


virahanka3(4)


# In[104]:


virahanka4(4)


# ## 4.8   A Sample of Python Libraries

# ### Matplotlib

# In[106]:


from numpy import arange
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black

def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind+c*width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind+width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()
    
genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
              (genre, word)
              for genre in genres
              for word in nltk.corpus.brown.words(categories=genre)
              if word in modals)
counts = {}
for genre in genres:
     counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)


# In[109]:


from matplotlib import use, pyplot
use('Agg')
pyplot.savefig('modals.png')
print('Content-Type: text/html')
print()
print('<html><body>')
print('<img src="modals.png"/>')
print('</body></html>')


# ### NetworkX

# In[114]:


import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn

def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)

def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G

def graph_draw(graph):
    nx.draw(graph,
         node_size = [16 * graph.degree(n) for n in graph],
         node_color = [graph.depth[n] for n in graph],
         with_labels = False)
    matplotlib.pyplot.show()

dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)


# ### csv

# In[119]:


import csv
f = open("lexicon.csv", "w")
f.write("lines with spaces \n more lines")
f.close()

input_file = open("lexicon.csv", "r")
for row in csv.reader(input_file):
    print(row)


# ### NumPy

# In[120]:


from numpy import array
cube = array([ [[0,0,0], [1,1,1], [2,2,2]],
                [[3,3,3], [4,4,4], [5,5,5]],
                [[6,6,6], [7,7,7], [8,8,8]] ])
cube[1, 1, 1]


# In[121]:


cube[2].transpose()


# In[122]:


cube[2, 1:]


# In[123]:


from numpy import linalg
a=array([[4,0], [3,-5]])
u,s,vt = linalg.svd(a)
u

