import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    T = np.zeros((N, L))
    bp = np.zeros((N, L), dtype=int)
    for i in range(N):
        for y in range(L):
            if i == 0:
                T[i, y] = emission_scores[i, y] + start_scores[y]
                # bp[i, y] is the start token
            else:
                T_candidates = emission_scores[i, y] + trans_scores[:, y] + T[i - 1, :]
                T[i, y] = np.max(T_candidates)
                bp[i, y] = np.argmax(T_candidates)

    final_scores = T[-1] + end_scores
    score = max(final_scores)
    y = [np.argmax(final_scores)]
    for i in reversed(range(1, N)):
        prev_y = bp[i, y[-1]]
        y.append(prev_y)

    return score, y[::-1]
