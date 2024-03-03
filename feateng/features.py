# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json
import spacy
ner = spacy.load('en_core_web_sm')

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

        # How many words long is the question?
        yield ("word", (len(run.split()) - 75) / 75)

        ftp = 0

        # How many characters long is the guess?
        if guess is None or guess=="":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))
            yield ("guess_words", log(len(guess.split())))


class CountFeature(Feature):

    #whether given string is a planidrome or not 
    def is_palindrome(string):
        return string == string[::-1]

    def __call__(self, question, run, guess, guess_history):
        
        #count the num of palindromes in q and guess
        question_split = run.split()
        guess_split = guess.split()

        pal_count_q = 0
        for word in question_split:
            if CountFeature.is_palindrome(word):
                pal_count_q += 1
        
        pal_count_g = 0 
        for word in guess_split:
            if CountFeature.is_palindrome(word):
                pal_count_g += 1 

        if pal_count_q != 0:
            yield("palindromes_question", log(pal_count_q))
        else:
            yield("palindromes_question", 0)

        if pal_count_g != 0:
            yield("palindromes_guess", (pal_count_g))
        else:
            yield("palindromes_guess", 0)

        


class EntityFeature(Feature):

    #try including POS info 

    def __call__(self, question, run, guess, guess_history):
        
        process = ner(guess)
        labels = []
        for item in process.ents:
            label = item.label_
            if(label == "GPE" or label == "LOC" or label == "NORP"):
                labels.append(label)

        if(len(labels) == 0):
            #yield("guess_entity", "NA")
            return 
        else: 
            yield("guess_entity", labels[0])


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
                self.counts[self.normalize(ii["page"])] += 1
                self.count_cat[self.normalize(ii["subcategory"])] += 1

    def __call__(self, question, run, guess, guess_history):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))
        yield ("subcat", log(1 + self.count_cat[self.normalize(question['subcategory'])]))


class CategoryFeature(Feature):
    def __call__(self, question, run, guess, guess_history):
        yield("sub", question['subcategory'])
        
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
