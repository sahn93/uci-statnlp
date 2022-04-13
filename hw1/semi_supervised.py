from speech import *
from classify import evaluate

from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import vstack

def self_train(Xu, Xl, yl, devX, devy, C=0.5, confident_cutoff=0.8):
    Xhat, yhat = Xl, yl
    num_iter = 0
    num_stall = 0
    curr_best = .0
    
    clss = []
    accs = []
    data_sizes = []
    
    while True:
        # Train
        num_iter += 1
        data_sizes.append(Xhat.shape[0])
        print(f"{num_iter}th train")
        print("Data size:", Xhat.shape, yhat.shape)
        cls = LogisticRegression(max_iter=10000, n_jobs=-1, C=C)
        cls.fit(Xhat, yhat)
        clss.append(cls)
        
        print("Evaluate Dev")
        acc = evaluate(devX, devy, cls)
        accs.append(acc)
        
        if acc > curr_best:
            print("new best score")
            curr_best = max(acc, curr_best)
            num_stall = 0
        else:
            num_stall += 1
            print(f"stall {num_stall} times")
            if num_stall >= 5:
                print(f"dev accuracy is not improving for {num_stall} iterations. Stop.")
                break

        # Predict
        print("Predicting unlabeled data with the previous model")
        yu_hat = cls.predict(Xu)
        confidents = cls.predict_proba(Xu).max(axis=1)

        # Expand Confident samples
        confident_Xu = Xu[confidents >= confident_cutoff]
        confident_yu_hat = yu_hat[confidents >= confident_cutoff]
        Xu = Xu[confidents < confident_cutoff]

        if confident_Xu.shape[0] == 0:
            print("Data size has converged")
            break
        
        print("Data added:", confident_Xu.shape)
        Xhat = vstack((Xhat, confident_Xu))
        yhat = np.concatenate((yhat, confident_yu_hat), axis=0)
            
    return clss, accs, data_sizes

def student_teacher(Xu, Xl, yl, devX, devy, C=0.5):
    # Train Teacher
    print(f"Train teacher model")
    print("Data size:", Xl.shape, yl.shape)
    teacher_cls = LogisticRegression(max_iter=10000, n_jobs=-1, C=C)
    teacher_cls.fit(Xl, yl)
    
    print("Evaluate Dev")
    teacher_acc = evaluate(devX, devy, teacher_cls)

    # Predict
    print("Predicting unlabeled data with the teacher model")
    yu_hat = teacher_cls.predict(Xu)
    confidents = teacher_cls.predict_proba(Xu).max(axis=1)

    # Get top K Confident samples for each classes
    K = 10000
    topk_Xus = [Xl]
    topk_y_hats = [yl]
    for label in np.unique(yl):
        X = Xu[yu_hat==label]
        y = yu_hat[yu_hat==label]
        print(X.shape, y.shape)
        K = min(X.shape[0], K)
        label_confs = confidents[yu_hat==label]
        
        ind = np.argpartition(label_confs, -K)[-K:]
        topk_Xus.append(X[ind])
        topk_y_hats.append(y[ind])

    Xhat = vstack(topk_Xus)
    yhat = np.concatenate(topk_y_hats, axis=0)
    
    # Train Student
    print(f"Train student model")
    print("Data size:", Xhat.shape, yhat.shape)
    student_cls = LogisticRegression(max_iter=10000, n_jobs=-1, C=C)
    student_cls.fit(Xhat, yhat)
    
    print("Evaluate Dev")
    student_acc = evaluate(devX, devy, student_cls)
            
    return student_cls, student_acc, teacher_cls, teacher_acc


from nltk.stem import WordNetLemmatizer

class Lemmatizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, sentence):
        return ' '.join([self.wnl.lemmatize(word) for word in sentence.split()])

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from speech import *
from classify import evaluate
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from os import listdir

def get_file_list(tsv_file):
    print(tsv_file)
    fnames = []
    with open(tsv_file, 'r') as f:
        for line in f:
            fname, label = line.strip().split('\t')
            fnames.append(f"data/speech/{fname}")
    return fnames

def get_unlabeled_file_list():
    lst = []
    dirname = 'data/speech/unlabeled'
    for fname in listdir(dirname):
        if ".txt" in fname:
            lst.append(f'{dirname}/{fname}')
    return lst

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
        # files = get_file_list("data/speech/train.tsv") + get_unlabeled_file_list()
        files = get_file_list("data/speech/train.tsv")
        self.tokenizer.train(files=files, trainer=trainer)

    def __call__(self, articles):
        return self.tokenizer.encode(articles).tokens


# Read file with Lemmatizer and BPE Tokenizer
bpe_tokenizer = BPETokenizer()
print("Reading data")
tarfname = "data/speech.tar.gz"
speech = read_files(tarfname, preprocessor=Lemmatizer(), tokenizer=bpe_tokenizer)
print(speech.trainX.shape)

print("Reading unlabeled data")
unlabeled = read_unlabeled(tarfname, speech)
print(unlabeled.X.shape)


# Train self-training with BPE tokenizer
print("Training self-train classifier")
clss, accs, data_sizes = self_train(unlabeled.X, speech.trainX, speech.trainy, speech.devX, speech.devy)


# Train with augmented data and BPE tokenizer
print("Training data augmented classifier")
student_cls, student_acc, teacher_cls, teacher_acc = student_teacher(
    unlabeled.X, speech.trainX, speech.trainy, speech.devX, speech.devy)

print("Evaluating")
evaluate(speech.trainX, speech.trainy, student_cls)
evaluate(speech.devX, speech.devy, student_cls)


# Read file with Lemmatizer and NLTK Tokenizer
from nltk import word_tokenize
print("Reading data")
tarfname = "data/speech.tar.gz"
nltk_speech = read_files(tarfname, preprocessor=Lemmatizer(), tokenizer=word_tokenize)
print(nltk_speech.trainX.shape)

print("Reading unlabeled data")
nltk_unlabeled = read_unlabeled(tarfname, nltk_speech)
print(nltk_unlabeled.X.shape)


# Train self-training with NLTK tokenizer
print("Training classifier")
nltk_clss, nltk_accs, nltk_data_sizes = self_train(
    nltk_unlabeled.X, nltk_speech.trainX, nltk_speech.trainy, nltk_speech.devX, nltk_speech.devy, C=0.5)


# Train with augmented data and NLTK tokenizer
print("Training classifier")
nltk_student_cls, nltk_student_acc, nltk_teacher_cls, nltk_teacher_acc = student_teacher(
    nltk_unlabeled.X, nltk_speech.trainX, nltk_speech.trainy, nltk_speech.devX, nltk_speech.devy)

print("Evaluating")
evaluate(nltk_speech.trainX, nltk_speech.trainy, nltk_student_cls)
evaluate(nltk_speech.devX, nltk_speech.devy, nltk_student_cls)