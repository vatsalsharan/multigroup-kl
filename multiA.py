import sklearn as sk
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def load_data(filename='mnist_784'):
    X, y = fetch_openml(filename, version=1, return_X_y=True)
    Y = np.zeros(len(y))
    for i in range(len(y)):
        Y[i] = int (y[i])
    return X, Y


def is_odd(num):
    return num % 2 != 0


# X is the images, Y is the label,
# return a sets of images P and Q that are biased mixtures of odd and even. And the kl distance between them.
# One set will have a 'bias' fraction of evens and other 1-bias.
# So if bias = 0.5 the two sets are coming from the same distribution.
# there are 34418 even images and 35582 odd images
def prepare_sets(X, Y, bias):
    x_odd = []
    x_even = []
    # partition the images to odd and even
    for i in range(len(Y)):
        if is_odd(Y[i]):
            x_odd.append(X[i])
        else:
            x_even.append(X[i])
    x_even = np.array(x_even)
    x_odd = np.array(x_odd)
    # permuting the arrays
    x_even = np.random.permutation(x_even)
    x_odd = np.random.permutation(x_odd)
    # create P, R by grabing from x_odd, x_even
    p_num_odd = int(bias * len(x_odd))
    r_num_odd = len(x_odd) - p_num_odd
    p_num_even = int((1 - bias) * len(x_even))
    r_num_even = len(x_even) - p_num_even
    r_even = r_num_even / (r_num_even + r_num_odd)
    p_even = p_num_even / (p_num_even + p_num_odd)
    r_odd = 1 - r_even
    p_odd = 1 - p_even
    kl = r_even * (np.log(r_even / p_even)) + r_odd * (np.log(r_odd / p_odd))
    P = np.zeros((p_num_odd + p_num_even, 784))
    R = np.zeros((r_num_odd + r_num_even, 784))
    P[:p_num_odd][:] = x_odd[:p_num_odd][:]
    P[-p_num_even:][:] = x_even[:p_num_even][:]
    R[:r_num_odd][:] = x_odd[-r_num_odd:][:]
    R[-r_num_even:][:] = x_even[-r_num_even:][:]
    P = np.random.permutation(P)
    R = np.random.permutation(R)
    return P, R, kl


def compute_advantage(clr, P, R, w_p, w_r, frac=0.5):
    p_size = int(frac*len(P))
    index_P = range(len(P))
    sample_index_P = np.random.choice(index_P, size=p_size, p=w_p / np.sum(w_p))
    sample_P = P[sample_index_P]
    sample_wp = w_p[sample_index_P]
    p_p = clr.predict(sample_P)
    p_succ = np.sum(p_p*sample_wp)/np.sum(sample_wp)
    r_size = int(frac * len(R))
    index_R = range(len(R))
    sample_index_R = np.random.choice(index_R, size=r_size, p=w_r / np.sum(w_r))
    sample_R = R[sample_index_R]
    r_p = clr.predict(sample_R)
    sample_rp = w_p[sample_index_R]
    r_succ = np.sum(r_p * sample_rp) / np.sum(sample_rp)
    print("success rate for R,P is: ", r_succ, p_succ)
    return(p_succ - r_succ)


def calc_advantage(clr, P, R, w_p, w_r):
    p_p = clr.predict(P)
    p_succ = np.sum(p_p * w_p) / np.sum(w_p)
    r_p = clr.predict(R)
    r_succ = np.sum(r_p * w_r) / np.sum(w_r)
    return (p_succ - r_succ)


def get_classifier():
    return RandomForestClassifier(max_depth=4, random_state=42)


#update the weights of P
def update_weights(clr, P, w_p, eta):
    update_vec = clr.predict(P)
    update_vec = 1 - update_vec
    update_vec = np.exp(eta * (update_vec))  # the misclassified items get higher weight
    return w_p * update_vec


def balance_sets(P_tr, R_tr, w_p, w_r):
    w_p1 = w_p * (np.sum(w_r) / np.sum(w_p))
    X_tr = np.vstack((P_tr, R_tr))
    w_tr = np.concatenate((w_p1, w_r))
    y_tr = np.concatenate((np.ones(len(P_tr)), np.zeros(len(R_tr))))
    a = np.arange(len(y_tr))
    a = np.random.permutation(a)
    X_tr = X_tr[a]
    y_tr = y_tr[a]
    w_tr = w_tr[a]
    return X_tr, y_tr, w_tr


# eps: the advantage required to add a classifier
# eta: the learning rate
# iter: the number of iterations
# clr_naker: a function that returns  classifier
def multiA_KL(P, R, iter=50, eps=0.05, eta=0.05, clr_maker=None):
    P = np.random.permutation(P)
    R = np.random.permutation(R)
    p_train = int(0.5 * len(P))
    r_train = int(0.5 * len(R))
    P_tr = P[:p_train]
    P_tst = P[p_train:]
    R_tr = R[:r_train]
    R_tst = R[r_train:]
    w_r_tr = np.ones(len(R_tr))
    w_r_tst = np.ones(len(R_tst))
    w_p_tr = np.ones(len(P_tr))
    w_p_tst = np.ones(len(P_tst))
    reg_list = []
    lambda_0 = np.log(np.sum(w_p_tst) / len(w_p_tst))
    for i in range(iter):
        print("iteration ", i)
        if clr_maker is None:
            clr = get_classifier()
        else:
            clr = clr_maker()
        X_tr, y_tr, w_tr = balance_sets(P_tr, R_tr, w_p_tr, w_r_tr)
        clr.fit(X_tr, y_tr, sample_weight=w_tr)
        advantage = calc_advantage(clr, P_tst, R_tst, w_p_tst, w_r_tst)
        print("advantage: ", advantage)
        if advantage > eps:
            print("updating weights")
            w_p_tst = update_weights(clr, P_tst, w_p_tst, eta)
            w_p_tr = update_weights(clr, P_tr, w_p_tr, eta)
            reg_list.append(clr)
            lambda_0 = np.average(w_p_tst)
        else:
            print("No classifier found")
            break
    return reg_list, lambda_0


class MaxEnt:

    def __init__(self, ):
        self.clr_list = []
        self.eta = 0
        self.lam = 0

    def fit(self, P, R, iter=50, eps=0.05, eta=0.05, clr_maker=None):
        self.eta = eta
        self.clr_list, self.lam = multiA_KL(P, R, iter, eps, eta, clr_maker)

    def compute_weight(self, point):
        w = np.ones(1)
        for clr in self.clr_list:
            w = update_weights(clr, point, w, self.eta)
        return w/self.lam

    def compute_KL(self, T):
        w = np.ones(len(T))
        for clr in self.clr_list:
            w = update_weights(clr, T, w, self.eta)
        w = w/self.lam
        return np.average(np.log2(w))


