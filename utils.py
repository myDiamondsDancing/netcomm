import numpy as np
import numpy.random as rnd


def Bernoulli_trial(p):
    return rnd.choice([True, False], p=[p, 1- p])

# def readint(
#     prompt,    # prompt string
#     condition  # correctness condition
# ):
#     temp = input(prompt)
#     try:
#         temp = int(temp)
#     except:
#         raise ValueError("'{}' is not integer".format(temp))
#     if not condition(temp):
#         raise ValueError(
#             "choosen number '{}' is not valid".format(temp))
#     return temp

def uncertainty(m):
    return np.array(m * [1.0 / m], float)

def define_dialogue_matrix(p, q):
    assert isinstance(p, float), \
           "the 1st parameter is not float"
    assert isinstance(q, float), \
           "the 2nd parameter is not float"
    assert 0 <= p <= 1, \
           "the 1st parameter is out of the segment [0, 1]"
    assert 0 <= q <= 1, \
           "the 2nd parameter is out of the segment [0, 1]"
    return np.array([p, 1 - p, 1 - q, q], float).reshape(2, 2)

def h(w):  # the normalised entropy of distribution 'w'
    h = float(0)
    for p in w:
        if p :
            h -= p * np.log2(p)
    return h

def input_non_empty(prompt):
    while True:
        temp = input(prompt)
        if temp:
            return temp
        else:
            print("Error!", end=' ')
