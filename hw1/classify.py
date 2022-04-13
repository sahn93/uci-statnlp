#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression()
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy", acc)
	return acc

def get_most_weighted_feat(logit_cls, speech, highest=True):
    count_vect = speech.count_vect
    le = speech.le
    speakers = le.inverse_transform([i for i in range(logit_cls.coef_.shape[0])])
    
    if highest:
        print("Highest weighted:")
        most_vec = np.argmax(logit_cls.coef_, axis=1)
    else:
        print("Lowest weighted:")
        most_vec = np.argmin(logit_cls.coef_, axis=1)
        
    most_mat = np.zeros((most_vec.size, logit_cls.coef_.shape[1]))
    most_mat[np.arange(most_vec.size), most_vec] = 1

    most_feats = [arr[0] for arr in count_vect.inverse_transform(most_mat)]
    d = {}
    for speaker, most_feat in zip(speakers, most_feats):
        print(f"{speaker}: {most_feat}")
    print()