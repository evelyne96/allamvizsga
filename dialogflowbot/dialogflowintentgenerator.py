import glob, os, json, re, random
from collections import OrderedDict
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
                    print(text)
                    tokens = [t for t in word_tokenize(text)]
                    tokens = pos_tag(tokens)
                    usersays.append(tokens)
            return usersays
    except Exception as e:
        print('Exception when reading: '+testfile+" "+str(e))
        return None

def writePhrases(fileName, sets_of_pharases):
    with open(fileName, 'w') as the_file:
        for phrases in sets_of_pharases:
            for p in phrases:
                the_file.write("%s\n" % p)
        the_file.close()

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

def getSynsetForWord(word, tag):
    try:
        if tag.startswith('NN'):
            return wordnet.synset(word+'.n.01')
        if tag.startswith('V'):
            return wordnet.synset(word+'.v.01')
        if tag.startswith('RB'):
            return None
    except:
        return None

def getSynonyms(word, tag):
    syns = [word]
    t = wnTag(tag)
    if t:
            word = wnl.lemmatize(word=word, pos=t)
            word_synset = getSynsetForWord(word, tag)
            syns = [s for s in wordnet.synsets(word, t)]
            if word_synset:
                syns = sorted(syns, key=lambda x: word_synset.wup_similarity(x), reverse=True)
            all_syns = []
            for s in syns:
                all_syns = all_syns + s.lemma_names()
            syns = list(OrderedDict.fromkeys(all_syns))
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
        'VBP' : conjugate(verb = word,tense= PRESENT, number = SINGULAR, person = 1),
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
    # data = readIntents(files[5])
    data = readIntents2()
    sentences = []

    if data != None :
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
            sentences.append((sentence, pattern))
    return sentences

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
    return check_pass_tense_trans(subj, pred, obj, pattern)

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

def random_new_sentence(sentence):
    paraphrases = []
    for _ in range(1,10):
        paraphrase = ""
        for (word, tag, syns, _) in sentence:
            if len(syns) > 4:
                next_word = syns[random.randint(0,4)]
            else:
                if len(syns) > 0:
                    next_word = syns[random.randint(0,len(syns) - 1)]
                else:
                  next_word = word  
            if tag.startswith('V'):
                next_word = getRightForm(next_word, tag)
            paraphrase = paraphrase + next_word + " "
        paraphrases.append(paraphrase)
    paraphrases = set(paraphrases)
    print(paraphrases)
    return paraphrases
    

def paraphrase():
    sentences = prepare()
    p = []
    for (sentence, pattern) in sentences:
        paraphrases = random_new_sentence(sentence)
        p.append(paraphrases)
        # if (isActivePattern(pattern, sentence)):
            # print('Active')
            # print(sentence)
            # paraphrases = random_new_sentence(sentence)
            # print(create_passive(sentence, pattern))
        # else:
            # print('Not active')
            # print(sentence) 
            # random_new_sentence(sentence)
    writePhrases("generated_phrases/see-ghost-more-passage.txt", p)

def readIntents2():
    data = []
    usersays = []    
    data = ["Discover what is in the passageway", "Go into the hole that opened up for you", "Discover the open passageway",
             "Follow the passageway to find_out what is inside it", "I want you to find_out what is in the passageway"]
    for tt in data:
         text = tt
         #elso sorban mondatokra kene bontani es azt szavakra es abbol csinalni egy listat
         print(text)
         tokens = [t for t in word_tokenize(text)]
         tokens = pos_tag(tokens)
         usersays.append(tokens)
    return usersays
paraphrase()