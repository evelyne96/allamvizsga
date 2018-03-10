import glob, os, json
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pattern.text.en import superlative ,comparative, pluralize, conjugate, PRESENT,SG, PAST, PARTICIPLE, SINGULAR

import nltk
nltk.download('averaged_perceptron_tagger')
wnl = WordNetLemmatizer()
# https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
# https://github.com/clips/pattern

def readIntents(fileToReadFrom):
    testfile = fileToReadFrom
    data = []
    try:
        with open(testfile) as data_file:
            usersays = []    
            data = json.load(data_file)
            for d in data:
                for t in d['data']:
                    text = t['text']
                    #elso sorban mondatokra kene bontani es azt szavakra es abbol csinalni egy listat
                    tokens = [t for t in word_tokenize(text)]
                    tokens = pos_tag(tokens)
                    usersays.append(tokens)
            return usersays
    except Exception as e:
        print('Exception when reading: '+testfile+" "+str(e))
        return None

def wnTag(tag):
    if tag.startswith('JJ'):
        return wordnet.ADJ
    if tag.startswith('NN'):
        return wordnet.NOUN
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('RB'):
        return wordnet.ADV
    return None

def getSynonyms(word, tag):
    syns = [word]
    t = wnTag(tag)
    if t:
            word = wnl.lemmatize(word=word, pos=t)
            syns = [s.lemmas()[0] for s in wordnet.synsets(word, t)]
            syns = set([l.name() for l in syns])
            if syns == None:
                syns = [(word, 1)]
            syns = [ getRightForm(w, tag) for w in syns]
    else: syns = [word]

    return syns

def getAntonyms(word, tag, word_before, tag_before):
    ants = []
    t = wnTag(tag)
    if t:
        if word_before == 'not' and tag.startswith('JJ'):
            word = wnl.lemmatize(word=word, pos=t)
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    if l.antonyms():
                        ants.append(l.antonyms()[0].name())
            if ants == None:
                ants = []
            ants = [ getRightForm(w, tag) for w in ants]
    
    return ants

def getRightForm(word, tag):
     return {
        'VB': word,
        'VBD': conjugate(verb = word,tense=PAST,number=SG),
        'VBG' : conjugate(verb = word, tens = PARTICIPLE, number = SG),
        'VBN' : conjugate(verb = word, tens = PARTICIPLE, number = SG),
        'VBP' : conjugate(verb = word,tense= PRESENT, number =SINGULAR),
        'VBZ' : conjugate(verb = word, tens = PRESENT, number = SG),
        'NN' : word,
        'NNP' : pluralize(word),
        'NNS' : word,
        'NNPS' : pluralize(word),
        'JJ' : word,
        'JJR' : comparative(word),
        'JJS' : superlative(word),
        'RB' : word,
        'RBR' : comparative(word),
        'RBS' : superlative(word),
    }[tag]


def paraphrase():
    path = os.path.join(os.getcwd(), 'intents', '*.json')
    files = glob.glob(path)
    data = readIntents(files[3])

    if data != None :
        # kell tokenizalni meg mondatonkent is es itt vegig menni a mondatokon kulon kulon
        for d in data:
            sentence = []
            (word_before, tag_before) = d[0]
            for index in range(len(d)):
                (word, tag) = d[index]
                antonyms = getAntonyms(word, tag, word_before, tag_before)
                synonyms = getSynonyms(word, tag)
                word_before, tag_before = d[index]

                sentence.append((word, synonyms, antonyms))

    print(str(sentence))

paraphrase()