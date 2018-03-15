import glob, os, json, re
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pattern.text.en import superlative ,comparative, pluralize, conjugate, PRESENT,SG,PAST, PARTICIPLE, SINGULAR, PROGRESSIVE, PLURAL

import nltk
nltk.download('averaged_perceptron_tagger')
wnl = WordNetLemmatizer()
# https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
# https://github.com/clips/pattern

subject = '(NNP-|NNS-|NNPS-|PRP-|NN-)'
actionDescriptor = '(RB-)'
determiner = '(IN-|DT-|JJ-|TO-)'
pres_predicate = '(VBP-|VBZ-)'
pas_smpl_predicate = '(VBD-)'
pres_cont_predicate = '((VBP-|VBZ-)(VBG-))'
pres_perfect_predicate = '((VBP-|VBZ-)(VBN-))'
pas_cont_predicate = '((VBD-)(VBG-))'

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
                    tokens = [t for t in word_tokenize('I was watching TV')]
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

def check_if_pres_pref(sentence):
    for (w, t, _, _) in sentence:
        if t == 'VBZ' or t == 'VBP':
            word = wnl.lemmatize(word=w, pos = wnTag(t))
            if word == 'have':
                return True
            else: 
                return False
    return False                

def isActivePattern(string, sentence):
    #the object look somewhat the same as the subject so we will use that on our pattern
    if (re.search(pres_perfect_predicate, string) == None):
        predicate = '('+pres_cont_predicate+'|'+pas_cont_predicate+'|'+pas_smpl_predicate+'|'+pres_predicate+'){1}'
        active_pat = re.compile('((PDT-|DT-|JJ-){1})?'+subject+'((CC-){1}'+subject+'{1})?'+
                             actionDescriptor+'?'+predicate+determiner+'?'+subject+'?')
        print(string)
        is_active = active_pat.match(string)
        return is_active != None
    else:
        if check_if_pres_pref(sentence):
            return True
        else:
            return False

def prepare():
    path = os.path.join(os.getcwd(),'vsbotintents', 'intents', '*.json')
    files = glob.glob(path)
    data = readIntents(files[3])

    if data != None :
        # kell tokenizalni meg mondatonkent is es itt vegig menni a mondatokon kulon kulon
        #ez eddig egy mondatra megy
        for d in data:
            sentence = []
            pattern = ""
            (word_before, tag_before) = d[0]
            for index in range(len(d)):
                (word, tag) = d[index]
                antonyms = getAntonyms(word, tag, word_before, tag_before)
                synonyms = getSynonyms(word, tag)
                word_before, tag_before = d[index]
                sentence.append((word, tag, synonyms, antonyms))
                pattern += tag+'-'
    return sentence, pattern

def create_passive(sentence, pattern):
    pred_was_found = False
    subj = ""
    pred = ""
    obj = ""
    for (w, t, _, _) in sentence:
        if t.startswith('V'):
            pred += w + " "
            pred_was_found = True
        elif pred_was_found:
            obj += w + " "
        else: 
            subj += w + " "
    print('S: ',subj,' P: ',pred,' O: ',obj)
    print(check_pass_tense_trans(subj, pred, obj, pattern))

def get_person(subj):
    pos_subj = pos_tag([s for s in word_tokenize(subj)])
    pattern = ""
    for (_, t) in pos_subj:
        pattern = pattern + t
    if re.search('NNS|NNPS', pattern) != None:
        return PLURAL, 3, subj
    else: 
        if re.search('PRP', pattern) != None:
            if re.search('They|they', subj):
                return PLURAL, 3, 'them'
            if  re.search('You|you', subj):
                return PLURAL, 3, 'you'
            if re.search('We|we', subj):
                return PLURAL, 3, 'us'
            if re.search('I|i', subj):
                return SINGULAR, 1 , 'me'
            if re.search('She|she', subj):
                return SINGULAR, 3, 'her'
            if re.search('He|he', subj):
                return SINGULAR, 3, 'him'
            
    return SINGULAR, 3, subj

def check_pass_tense_trans(subj, pred, obj, pattern):
    number, person, s = get_person(subj)
    number, person, _ = get_person(obj)

    if re.search(pres_cont_predicate, pattern) or re.search(pres_perfect_predicate, pattern) or re.search(pas_cont_predicate, pattern):
        
        word_to_use,pred = wnl.lemmatize(word=pred.split(' ',1)[0].strip(), pos=wnTag('V')) ,pred.split(' ', 1)[1]

        if re.search(pas_cont_predicate, pattern):
            start = conjugate(word_to_use, PAST, person, number)
        else:
            start = conjugate(word_to_use, PRESENT, person, number)

        pred = start+" being "+" "+conjugate(wnl.lemmatize(word=pred.strip(), pos=wnTag('V')), PAST, aspect = PROGRESSIVE)
        return obj + " "+pred+" by "+ s
    else: 
            if re.search(pas_smpl_predicate, pattern):
                start = conjugate('be', PAST, person, number)
            else:
                start = conjugate('be', PRESENT, person, number)
                
            pred = start+ " " + conjugate(wnl.lemmatize(word=pred.strip(), pos=wnTag('V')), PAST, aspect = PROGRESSIVE)
            return obj + " " + pred + " by "+s


                
def paraphrase():
    sentence, pattern = prepare()
    if (isActivePattern(pattern, sentence)):
        print('Active')
        return create_passive(sentence, pattern)
    else:
        print('Not active')
        return sentence 

paraphrase()