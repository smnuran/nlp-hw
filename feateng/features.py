# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from numpy import mean
import numpy as np 
import gzip
import json
import spacy
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
ner = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history):
        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history):
        # How many characters long is the question?
        yield ("char", (len(run) - 450) / 450)
        # if run is None or run == "":
        #     yield("question_char", -1)
        #     yield("question_word", -1)
        # else: 
        #     yield("question_char", log(1+ len(run)))
        #     yield("question_word", log(1+ len(run.split())))

        # How many words long is the question?
        yield ("word", (len(run.split()) - 75) / 75)



        # How many characters long is the guess?
        if guess is None or guess=="":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))
            yield ("guess_words", log(1 + len(guess.split())))
            yield ("ratio", log(len(guess)/len(run)))


class CountFeature(Feature):

    #whether given string is a planidrome or not 
    def is_palindrome(string):
        return string == string[::-1]

    def __call__(self, question, run, guess, guess_history):
        
        # #count the num of palindromes in q and guess
        # question_split = run.split()
        # guess_split = guess.split()

        # pal_count_q = 0
        # for word in question_split:
        #     if CountFeature.is_palindrome(word):
        #         pal_count_q += 1
        
        # pal_count_g = 0 
        # for word in guess_split:
        #     if CountFeature.is_palindrome(word):
        #         pal_count_g += 1 

        # if pal_count_q != 0:
        #     yield("palindromes_question", (pal_count_q))
        # else:
        #     yield("palindromes_question", 0)

        # if pal_count_g != 0:
        #     yield("palindromes_guess", log(pal_count_g))
        # else:
        #     yield("palindromes_guess", 0)
            
        #count special symbols 
        symbs = ['[', ']', '(', ')', '_', '-', ',', '.']
        count = 0
        for char in guess:
            if char in symbs:
                count += 1
        
        # yield("symbols", log(1+ count))

        count = 0 
        for char in run:
            if char in symbs:
                count += 1 
        
        yield("symbols_question", log(count + 1))

        
#has parenthesis?
            
#do something with tokens : cosine sim 
            
#if guess in answer prompt
            
# "for 10 points" we have to buzz if we see this. the question is ending
            
class ExperimentFeature(Feature):
    def __call__(self, question, run, guess, guess_history):
        
        #if guess has parenthesis 
        open = guess.find("(")
        close = guess.find(")")
        if(open != -1 and close != -1):
            info = guess[open+1:close]


            question_spacy = ner(run)
            guess_spacy = ner(guess)
            info_spacy = ner(info)

            cos = guess_spacy.similarity(question_spacy)

            cos_info = info_spacy.similarity(question_spacy)


            yield("guess_ques_sim", cos)
            yield("info_ques_sim", cos_info)


            # if info == "Star Trek":
            #     yield("paren_info", info)
            yield("has_paren", True)
        else:
            yield("has_paren", False)

        #if guess appears in answer prompt field
        ans_p = question['answer_prompt'].lower()
        guess_low = guess.lower()
        if ans_p.find(guess_low) != -1:
            yield("guess_in_ap", True)

        for word in guess_low:
            if run.lower().find(word) != -1:
                yield("guess_in_question", 'True')


        # near the end of question "for _ points "
        run_low = run.lower()
        if (run_low.find('name this') != -1 and run_low.find('points') != -1):
            yield("run_near_end", True) 
        

    

class EntityFeature(Feature):
    effective_pos = ['JJ', 'NN', 'NNP']

    #try including POS info 
    def get_pos(text):
        tok = nltk.word_tokenize(text)
        return nltk.pos_tag(tok)

    def __call__(self, question, run, guess, guess_history):
        
        process = ner(guess)
        labels = []
        for item in process.ents:
            label = item.label_
            if(label == "LOC" or label == "NORP" or label == "GPE"): # or GPE
                labels.append(label)

        if(len(labels) != 0): 
            yield("guess_entity", labels[0])

        if guess != None or guess != "":
            guess_pos = EntityFeature.get_pos(guess)
            if (len(guess_pos) != 0):
                guess_pos = guess_pos[0][1]
                if(guess_pos in EntityFeature.effective_pos):
                    yield("guess_pos", guess_pos)


class FrequencyFeature(Feature):
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name 
        self.counts = Counter()
        self.count_cat = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
            for ii in questions:
                self.counts[self.normalize(ii["page"])] += 1 #what about the words of the guess
                #self.count_cat[self.normalize(ii["subcategory"])] += 1

    def __call__(self, question, run, guess, guess_history):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))
        #yield ("subcat", log(1 + self.count_cat[self.normalize(question['subcategory'])]))


class CategoryFeature(Feature):
    def __call__(self, question, run, guess, guess_history):
        subcat = question['subcategory']
        cat = question['category']

        effective = ['Science Physics', 'Literature Classical', 'History World', 'Philosophy', 'Geography', 'History']

        if(subcat in effective):
            yield("sub", subcat)
       # if(cat in effective):
            #yield("cat", cat)

        #yield("tournament", question['tournament'])
        
class SentimentFeature(Feature):
    def __call__(self, question, run, guess, guess_history):

        sa = SentimentIntensityAnalyzer()

        #score_run = sa.polarity_scores(run)
        score_guess = sa.polarity_scores(guess)
        #score_subcat = sa.polarity_scores(question['subcategory'])

        #yield('question', score_run['compound'])
        #yield('question_neutral', score_run['neu'])
        #yield('guess', score_guess['compound'])
        yield('guess_neutral', score_guess['neu'])
        yield('guess_pos', score_guess['pos'])
        yield('guess_neg', score_guess['neg'])
        #yield('subcat', score_subcat['neu'])
        
        
class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess, guess_history):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess, guess_history):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from params import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    add_guesser_params(parser)
    add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
