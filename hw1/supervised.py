from speech import *
from classify import evaluate


# Default classifier
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname)
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# Default + Bigram
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, ngram_range=(1,2))
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# Default + Bigram (max_feature=10000)
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, max_features=10000, ngram_range=(1,2))
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# Default + TF-IDF feature
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, max_features=10000)
print(speech.trainX.shape)


from sklearn.feature_extraction.text import TfidfTransformer

print("Transforming TF-IDF")
transformer = TfidfTransformer()
tfidfX = transformer.fit_transform(speech.trainX)
tfidf_devX = transformer.fit_transform(speech.devX)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(tfidfX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(tfidfX, speech.trainy, cls))
    dev_accs.append(evaluate(tfidf_devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# + Lemmatization
from nltk.stem import WordNetLemmatizer

class Lemmatizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, sentence):
        return ' '.join([self.wnl.lemmatize(word) for word in sentence.split()])

print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, preprocessor=Lemmatizer())
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# + NLTK Tokenizer
from nltk import word_tokenize          

print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, tokenizer=word_tokenize)
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=10000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# Lemmatizer + NLTK Tokenizer
from nltk import word_tokenize

print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, preprocessor=Lemmatizer(), tokenizer=word_tokenize)
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# BPE Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents

from speech import *
from classify import evaluate

def get_file_list(tsv_file):
    print(tsv_file)
    fnames = []
    with open(tsv_file, 'r') as f:
        for line in f:
            fname, label = line.strip().split('\t')
            fnames.append(f"data/speech/{fname}")
    return fnames

class BPETokenizer(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        print("Training BPE with training data:")
        files = get_file_list("data/speech/train.tsv")
        self.tokenizer.train(files=files, trainer=trainer)

    def __call__(self, articles):
        return self.tokenizer.encode(articles).tokens

bpe_tokenizer = BPETokenizer()

print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, tokenizer=bpe_tokenizer)
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))
print(dev_accs)
print(max(dev_accs))


# Lemmatizer + BPE Tokenizer
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, preprocessor=Lemmatizer(), tokenizer=bpe_tokenizer)
print(speech.trainX.shape)

train_accs = []
dev_accs = []
Cs = [0.1*i for i in range(1, 11)]
for C in Cs:
    print("Training classifier")
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(max_iter=1000, n_jobs=-1, C=C)
    cls.fit(speech.trainX, speech.trainy)

    print("Evaluating")
    train_accs.append(evaluate(speech.trainX, speech.trainy, cls))
    dev_accs.append(evaluate(speech.devX, speech.devy, cls))

print(dev_accs)
print(max(dev_accs))