import string
import spacy


class SpellChecker():

    def __init__(self, max_distance, channel_model=None, language_model=None):
        self.nlp = spacy.load("en", pipeline=["tagger", "parser"])

    def load_channel_model(self, fp):
        self.channel_model = pickle.load(fp)

    def load_language_model(self, fp):
        self.language_model = pickle.load(fp)

    def bigram_score(self, prev_word, focus_word, next_word):
        prevFocusScore = self.language_model.bigram_prob(prev_word, focus_word)
        focusNextScore = self.language_model.bigram_prob(focus_word, next_word)
        return (prevFocusScore + focusNextScore)/2

    def unigram_score(self, word):
        return self.language_model.unigram_prob(word)

    def cm_score(self, error_word, corrected_word):
        return self.channel_model.prob(error_word, corrected_word)

    def inserts(self, word):
        '''
            Takes in word and return a list of words that are within one insert of word
        '''
        # Insert every letter
        possibleWords = []
        for letter in string.ascii_lowercase:
            # Every possible position
            for i in range(len(word) + 1):
                # Check if the resulting word is a word
                testWord = word[:i] + letter + word[i:]
                if language_model.__contains__(testWord):
                    possibleWords.append(testWord)
        return possibleWords

    def deletes(self, word):
        return
