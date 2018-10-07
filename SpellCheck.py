import string
import spacy
import pickle
from LanguageModel import LanguageModel
from EditDistance import EditDistanceFinder


class SpellChecker():

    def __init__(self, max_distance, channel_model=None, language_model=None):
        self.nlp = spacy.load('en', pipeline=["tagger", "parser"])
        self.max_distance = max_distance
        # self.load_channel_model(channel_model)
        # self.load_language_model(language_model)

    def load_channel_model(self, fp):
        self.channel_model = EditDistanceFinder()
        self.channel_model.load(fp)

    def load_language_model(self, fp):
        self.language_model = LanguageModel()
        self.language_model.load(fp)

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
                if self.language_model.__contains__(testWord):
                    possibleWords.append(testWord)
        return possibleWords

    def deletes(self, word):
         # Delete every letter
        possibleWords = []
        for i in range(len(word) + 1):
            # Check if the resulting word is a word
            testWord = word[:i] + word[i+1:]
            if self.language_model.__contains__(testWord):
                possibleWords.append(testWord)
        return possibleWords

    def substitutions(self, word):
         # Substitute every letter
        possibleWords = []
        for letter in string.ascii_lowercase:
            # At every possible position
            for i in range(len(word) + 1):
                # Check if the resulting word is a word
                testWord = word[:i] + letter + word[i + 1:]
                if self.language_model.__contains__(testWord):
                    possibleWords.append(testWord)
        return possibleWords

    def generate_candidates(self, word):
        '''
            Takes in a candidate word and returns words that are within self.max_distance edits of word
        '''
        for i in range(1, self.max_distance + 1):
            if i == 1:
                candidateWords = self.inserts(
                    word) + self.deletes(word) + self.substitutions(word)
            else:
                newWords = []
                for currentWord in candidateWords:
                    newWords += self.inserts(
                        currentWord) + self.deletes(currentWord) + self.substitutions(currentWord)
                candidateWords += newWords
        # Get rid of duplicates
        return list(set(candidateWords))

    def check_sentence(self, sentence, fallback=False):
        returnList = []
        for i in range(len(sentence)):
            if i == 0 and i == len(sentence) - 1:
                prevWord = '<s>'
                nextWord = '</s>'
            elif i == 0:
                prevWord = '<s>'
                nextWord = sentence[i+1]
            elif i == len(sentence) - 1:
                nextWord = '</s>'
                prevWord = sentence[i-1]
            else:
                prevWord = sentence[i-1]
                nextWord = sentence[i+1]
            word = sentence[i]
            # If it's in the language model, add just that word
            if self.language_model.__contains__(word):
                returnList.append([word])
            else:
                # Get all the candidates for that word
                candidates = self.generate_candidates(word)
                candidateList = []
                if candidates == [] and fallback:
                    candidateList = [word]
                else:
                    for candidate in candidates:
                        unigramScore = self.unigram_score(candidate)
                        bigramScore = self.bigram_score(
                            prevWord, candidate, nextWord)
                        languageScore = (0.5*unigramScore) + \
                            (0.5 * bigramScore)
                        candidateScore = languageScore + \
                            self.cm_score(word, candidate)

                        candidateList.append([candidate, candidateScore])

                    # Sort the list by the second element
                    candidateList.sort(key=lambda x: x[1], reverse=True)
                    # Remove the second element, and append
                    candidateList = [x[0] for x in candidateList]
                returnList += [candidateList]

        return returnList

    def check_text(self, text, fallback=False):
        '''
        take a string as input, tokenize and sentence segment it with spacy, and then return the concatenation of the result of calling check_sentence on all of the resulting sentence objects.
        '''
        tokens = self.nlp(text)
        sentences = list(tokens.sents)

        processedSentences = []
        for sentence in sentences:
            # Convert sentence into list of lowercase words
            wordList = sentence.text.split()
            wordList = [x.lower() for x in wordList]
            processedSentences.append(self.check_sentence(wordList, fallback))

        return processedSentences

    def autocorrect_sentence(self, sentence):
        '''
         take a tokenized sentence (as a list of words) as input, call check_sentence on the sentence with fallback=True, and return a new list of tokens where each non-word has been replaced by its most likely spelling correction.
        '''
        corrections = self.check_sentence(sentence, fallback=True)
        return [x[0] for x in corrections]

    def autocorrect_line(self, line):
        '''
             take a string as input, tokenize and segment it with spacy, and then return the concatenation of the result of calling autocorrect_sentence on all of the resulting sentence objects.
        '''

        tokens = self.nlp(line)
        sentences = list(tokens.sents)

        processedSentences = []
        for sentence in sentences:
            # Convert sentence into list of lowercase words
            wordList = sentence.text.split()
            if len(wordList) == 0:
                continue
            wordList = [x.lower() for x in wordList]
            processedSentences.append(self.autocorrect_sentence(wordList))

        return processedSentences

    def suggest_sentence(self, sentence, max_suggestions):
        '''
            take a tokenized sentence (as a list of words) as input, call check_sentence on the sentence, and return a new list where:
            Real words are just strings in the list
            Non-words are lists of up to max_suggestions suggested spellings, ordered by your model’s preference for them.
        '''
        sentenceCorrections = self.check_sentence(sentence)

        returnList = []
        for word in sentenceCorrections:
            if len(word) == 1:
                returnList += word
            else:
                returnList.append(word[:max_suggestions])

        return returnList

    def suggest_text(self, text, max_suggestions):
        '''
            take a string as input, tokenize and segment it with spacy, and then return the concatenation of the result of calling suggest_sentence on all of the resulting sentence objects
        '''
        tokens = self.nlp(text)
        sentences = list(tokens.sents)

        processedSentences = []
        for sentence in sentences:
            # Convert sentence into list of lowercase words
            wordList = sentence.text.split()
            wordList = [x.lower() for x in wordList]
            # Get rid of the period
            if wordList[-1][-1] == '.':
                wordList[-1] = wordList[-1][:-1]
            processedSentences.append(
                self.suggest_sentence(wordList, max_suggestions))

        return processedSentences
