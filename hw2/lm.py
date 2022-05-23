#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        numOOV = self.get_num_oov(corpus)
        return pow(2.0, self.entropy(corpus, numOOV))

    def get_num_oov(self, corpus):
        vocab_set = set(self.vocab())
        words_set = set(itertools.chain(*corpus))
        numOOV = len(words_set - vocab_set)
        return numOOV

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

class Ngram(LangModel):
    """
    Smoothing:
    - Laplace
    - Backoff
    - Interpolation (avg)
    """
    def __init__(self, ngram_size, laplace, backoff, interpolat, unk_prob=0.0001):
        self.ngram_size = ngram_size
        if backoff or interpolat:
            self.models = [dict() for i in range(ngram_size)]
        else:
            self.models = [dict()]
        self.lambdas = interpolat
        self.lunk_prob = log(unk_prob, 2)
        self.vocabulary = {'END_OF_SENTENCE'}
        self.laplace = laplace

    def inc_word(self, prefix, word, i):
        prefix = ' '.join(prefix)
        model = self.models[i]

        if prefix not in model:
            model[prefix] = dict()

        if word in model[prefix]:
            model[prefix][word] += 1.0
        else:
            model[prefix][word] = 1.0

    # required, update the model when a sentence is observed
    def _fit_sentence(self, sentence, n):
        sentence = ['<s>'] * (n - 1) + sentence
        model_idx = self.ngram_size - n

        for i in range(len(sentence) - n + 1):
            ngram = sentence[i:i+n]
            self.inc_word(ngram[:-1], ngram[-1], model_idx)

        if n == 1:
            self.inc_word([], 'END_OF_SENTENCE', model_idx)
        else:
            self.inc_word(sentence[-(n-1):], 'END_OF_SENTENCE', model_idx)

    def fit_sentence(self, sentence):
        self.vocabulary |= set(sentence)
        for i in range(len(self.models)):
            self._fit_sentence(sentence, self.ngram_size - i)

    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self):
        """Normalize and convert to log2-probs."""
        for model in self.models:
            for prefix in model:
                tot = 0.0
                for word in model[prefix]:
                    tot += model[prefix][word]
                ltot = log(tot + self.laplace * len(self.vocabulary), 2)

                for word in model[prefix]:
                    model[prefix][word] = log(model[prefix][word] + self.laplace, 2) - ltot
        
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV):
        # OOV
        if word not in self.vocabulary:
            return self.lunk_prob - log(numOOV, 2)

        previous = ['<s>'] * (self.ngram_size - 1) + previous
        
        leps = log(1e-6, 2)

        if len(self.lambdas) > 0:
            lsum = 0.0
            for i, model in enumerate(self.models):
                n = self.ngram_size - i
                if n == 1:
                    prefix = ''
                else:
                    prefix = ' '.join(previous[-(n-1):])

                if prefix in model and word in model[prefix]:
                    lsum += model[prefix][word] * self.lambdas[i]
                else:
                    lsum += leps * self.lambdas[i]
            return lsum

        for i, model in enumerate(self.models):
            n = self.ngram_size - i
            if n == 1:
                prefix = ''
            else:
                prefix = ' '.join(previous[-(n-1):])
            if prefix in model and word in model[prefix]:
                return model[prefix][word]
        
        # Not OOV but no such ngram -> return with almost zero possibility
        return leps

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self.vocabulary