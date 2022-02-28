import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class BranchingProgram:
    # width_array: a numpy array with the set of left boundaries of the cells, so starting at 0.
    # maxDepth: a maximum number of depth allowed for the branching program
    # epsilon: The min advantage over half, required for splitting.
    # data_P, data_R: The two data sets
    # classifier: a method that returns a classifier
    def __init__(self, width_array, max_depth, epsilon, classifier, reweigh=0):
        self.width = np.size(width_array)
        self.sets = width_array
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.classifier = classifier
        self.reweigh = reweigh
        self.nodes = []  # list of lists. Self.nodes[i] is the i'th layer of the program.
        self.middle = np.searchsorted(self.sets, 1) - 1 # finds the height corresponding to 1.
        self.root = Node(level=0, height=self.middle, cols=0, program=self)
        #self.nodes.append([self.root])
        self.dim = 0 # dimension of the data
        self.depth = 0
        self.was_fitted = False
        self.predicted_prob = []

    def make_layer(self, level, col):
        layer = []
        for i in range(self.width):
            layer.append(Node(level, i, col, self))
        return layer

    def start_afresh(self):
        self.nodes = []
        self.depth = 0
        self.root = Node(level=0, height=self.middle, cols=0, program=self)

    def predicted_kl(self, reverse = False):
        total_p = 0
        total_r = 0
        for node in self.nodes[self.depth - 1]:
            total_p = total_p + len(node.pts_test_p)
            total_r = total_r + len(node.pts_test_r)
        print("total r,p is ", total_r, total_p)
        res = 0
        #for node in self.nodes[self.depth - 1]:
        for i in range(len(self.nodes[self.depth - 1])):
            node = self.nodes[self.depth - 1][i]
            p_i = len(node.pts_test_p) / total_p
            r_i = len(node.pts_test_r) / total_r
            if p_i == 0 and r_i == 0:
                continue
            if p_i == 0 or r_i == 0:
                print("Seems distributions have different support. Computing only for common support")
                continue
            if reverse:
                res = res + p_i * np.log2(p_i / self.predicted_prob[i])
            else:
                res = res + r_i * np.log2(r_i / self.predicted_prob[i])
                print("adding ", r_i * np.log2(r_i / self.predicted_prob[i]), " to the kl")
        return res

    def predict_kl_pair(self, data_p, data_r, reverse = False):
        self.predict(data_r)
        depth = self.depth
        width = self.width
        prob_r = np.zeros(width)
        for i in range(width):
            prob_r[i] = self.predicted_prob[i]

        self.predict(data_p)
        prob_p = np.zeros(width)
        for i in range(width):
            prob_p[i] = self.predicted_prob[i]

        res = 0
        for i in np.arange(width):
            p_i = prob_p[i]
            r_i = prob_r[i]
            if p_i == 0 and r_i == 0:
                continue
            if p_i == 0 or r_i == 0:
                print("Seems distributions have different support. Computing only for common support")
                continue
            if reverse:
                res = res + p_i * np.log2(p_i / self.predicted_prob[i])
            else:
                res = res + r_i * np.log2(r_i / self.predicted_prob[i])
                print("adding ", r_i * np.log2(r_i / self.predicted_prob[i]), " to the kl")
        
        return res

    # Computed the KL divergense kl(r,p).
    # reverse: when set to True compute KL(p,r).
    def compute_kl(self, reverse = False):
        total_p = 0
        total_r = 0
        for node in self.nodes[self.depth - 1]:
            total_p = total_p + len(node.pts_test_p)
            total_r = total_r + len(node.pts_test_r)
        print("total r,p is ", total_r, total_p)
        res = 0
        for node in self.nodes[self.depth - 1]:
            p_i = len(node.pts_test_p)/total_p
            r_i = len(node.pts_test_r) / total_r
            if p_i == 0 and r_i == 0:
                continue
            if p_i == 0 or r_i == 0:
                print("Seems distributions have different support. Computing only for common support")
                continue
            if reverse:
                res = res + p_i * np.log2(p_i / r_i)
            else:
                res = res + r_i * np.log2(r_i / p_i)
                print("adding ", r_i * np.log2(r_i / p_i), " to the kl")
        return res

    def fit(self, data_p, data_r):
        self.was_fitted = True
        if len(np.shape(data_p)) != 2 or len(np.shape(data_r)) != 2:
            print("Data needs to be a matrix")
            return
        if np.shape(data_p) != np.shape(data_r):
            print("Error: need to start with same number of samples from R and P")
            return
        self.start_afresh()
        lines, col = np.shape(data_p)
        self.dim = col
        y = np.zeros(lines)
        X_train, X_test, _, _ = train_test_split(data_p, y, test_size = 0.5, random_state = 42)
        self.root.pts_train_p = X_train
        self.root.pts_test_p = X_test
        y = np.zeros(data_r[:, 0].size)
        X_train, X_test, _, _ = train_test_split(data_r, y, test_size=0.5, random_state=42)
        self.root.pts_train_r = X_train
        self.root.pts_test_r = X_test
        self.nodes.append([self.root])
        self.nodes.append(self.make_layer(1, col))
        self.nodes[0][0].split()
        keep_splitting  = True
        curr_level = 0
        while keep_splitting and curr_level < self.max_depth:
            keep_splitting = False;
            for item in self.nodes[curr_level]:
                keep_splitting = keep_splitting or item.did_split
            if keep_splitting:
                curr_level = curr_level + 1
                self.nodes.append(self.make_layer(curr_level + 1, col))
                for item in self.nodes[curr_level]:
                    item.split()
        self.depth = len(self.nodes)

    def predict(self, pts):
        num_points, dim_points = np.shape(pts)
        if dim_points != self.dim:
            print("Predicted points dimension doesn't match data")
            return []
        if num_points == 0:
            print("no points to predict")
            return []
        self.clear_predicted_points()
        self.nodes[0][0].pts_predicted = pts
        for level in range(self.depth - 1):
            for node in self.nodes[level]:
                node.predict()
        self.predicted_prob = np.zeros(self.width)
        for i in range(self.width):
            self.predicted_prob[i] = len(self.nodes[self.depth - 1][i].pts_predicted)/num_points

    def get_weight(self, pt):
        pt = pt.reshape(1, -1)
        num_points, dim_points = np.shape(pt)
        if num_points > 1:
            print("function takes one point at a time")
            return 0
        if dim_points != self.dim:
            print("dimension of point doesn't match fitted data")
            return 0
        curr_height = 0
        for level in range(self.depth - 1):
            if self.nodes[level][curr_height].was_fitted:
                passed = self.nodes[level][curr_height].classifier.predict(pt)
                if passed == 1:
                    curr_height = self.nodes[level][curr_height].next_layer_1
                else:
                    curr_height = self.nodes[level][curr_height].next_layer_0
        return self.nodes[self.depth - 1][curr_height].ratio

    def kl_sub(self, P, R):
        self.clear_predicted_points()
        self.predict(P)
        p_prob = self.predicted_prob
        self.clear_predicted_points()
        self.predict(R)
        r_prob = self.predicted_prob
        kl = kl_rev = 0 # kl = KL(P,R), kl_rev = KL(R,P)
        for i in range(len(p_prob)):
            if p_prob[i] == 0 and r_prob[i] == 0:
                continue
            if p_prob[i] == 0 or r_prob[i] == 0:
                print("Seems distributions have different support. Computing only for common support")
                continue
            kl = kl + p_prob[i] * np.log2(p_prob[i] / r_prob[i])
            kl_rev = kl_rev + r_prob[i] * np.log2(r_prob[i] / p_prob[i])
        return kl, kl_rev

    # clear all predicted points so new points could be predicted
    def clear_predicted_points(self):
        for level in self.nodes:
            for node in level:
                node.pts_predicted = np.array([], dtype=np.float64).reshape(0, self.dim)

    @staticmethod
    def get_classifier():
        return RandomForestClassifier(max_depth=5, n_estimators=15)
        #return AdaBoostClassifier()
        #return DecisionTreeClassifier(max_depth=5)
        #return MLPClassifier(alpha=1, max_iter=1000) # can't use becasue doesn't take sample weight
        #return GaussianNB()


class Node:
    # level: the depth in the branching program
    # height: which node is it in the level
    # cols: number of columns in the data
    # program: a pointer back to the branching program
    def __init__(self, level, height, cols, program):
        self.level = level
        self.height = height
        self.program = program
        self.pts_train_p = np.array([], dtype=np.float64).reshape(0, cols)
        self.pts_train_r = np.array([], dtype=np.float64).reshape(0, cols)
        self.pts_test_p = np.array([], dtype=np.float64).reshape(0, cols)
        self.pts_test_r = np.array([], dtype=np.float64).reshape(0, cols)
        self.pts_predicted = np.array([], dtype=np.float64).reshape(0, cols)

        self.ratio = self.program.sets[height]
        self.classifier = program.classifier()
        self.reweigh = program.reweigh
        self.did_split = False # indicates whether a split was successful
        # pointers to the nodes in the next layer. Default is to stay in the level. Will be updated at "split"
        self.next_layer_0 = height
        self.next_layer_1 = height
        self.was_fitted = False

    def __str__(self):
        str_val = "my level is  "
        str_val += str(self.level)
        str_val += "and my height is "
        str_val += str(self.height)
        return str_val

    # trains the classifier, and if successful, splits the points in the node and moves them to the next level.
    def split(self):
        # print("in split of node ", self.level, self.height)
        if self.pts_train_r.size == 0 or self.pts_train_p.size == 0:
            self.no_split()
            return
        elif self.pts_test_r.size == 0 or self.pts_test_p.size == 0:
            self.no_split()
            return
        else:
            self.ratio = self.pts_train_p.size / self.pts_train_r.size
        if self.reweigh == 1:
            x_train, y_train, weights_train = self.prepare_points_weigh(self.pts_train_p, self.pts_train_r)
            # print("calling fit on ",x_train, "and", y_train)
            # print("calling fit in node", self.level, self.height)
            # print("size of things to fit:", x_train.size)
            # print("running", type(self.classifier))
            self.classifier.fit(X=x_train, y=y_train, sample_weight=weights_train)
            self.was_fitted = True
            x_test, y_test, weights_test = self.prepare_points_weigh(self.pts_test_p, self.pts_test_r)
            # print("calling score on ", x_test, "and", y_test)
            score = self.classifier.score(X=x_test, y=y_test, sample_weight=weights_test)
            # print("score is ", score)
        else:
            x_train, y_train = self.prepare_points_sample(self.pts_train_p, self.pts_train_r)
            # print("calling fit on ",x_train, "and", y_train)
            self.classifier.fit(X=x_train, y=y_train)
            self.was_fitted = True
            x_test, y_test = self.prepare_points_sample(self.pts_test_p, self.pts_test_r)
            # print("calling score on ", x_test, "and", y_test)
            score = self.classifier.score(X=x_test, y=y_test)
            # print("score is ", score)
        if score > 0.5 + self.program.epsilon:
            self.did_split = True
            if len(self.pts_train_p) == 0:
                p_train_0 = 0
                p_train_1 = 0
            else:
                prediction_p_train = self.classifier.predict(self.pts_train_p)
                p_train_0 = self.pts_train_p[prediction_p_train == 0]
                p_train_1 = self.pts_train_p[prediction_p_train == 1]

            if len(self.pts_train_r) == 0:
                r_train_0 = 0
                r_train_1 = 0
            else:
                prediction_r_train = self.classifier.predict(self.pts_train_r)
                r_train_0 = self.pts_train_r[prediction_r_train == 0]
                r_train_1 = self.pts_train_r[prediction_r_train == 1]

            if len(self.pts_test_p) == 0:
                p_test_0 = 0
                p_test_1 = 0
            else:
                prediction_p_test = self.classifier.predict(self.pts_test_p)
                p_test_0 = self.pts_test_p[prediction_p_test == 0]
                p_test_1 = self.pts_test_p[prediction_p_test == 1]

            if len(self.pts_test_r) == 0:
                r_test_0 = []
                r_test_1 = []
            else:
                prediction_r_test = self.classifier.predict(self.pts_test_r)
                r_test_0 = self.pts_test_r[prediction_r_test == 0]
                r_test_1 = self.pts_test_r[prediction_r_test == 1]

            if r_test_0.size == 0:
                ratio_0 = 100
            else:
                ratio_0 = p_test_0.size / r_test_0.size
            self.next_layer_0 = np.searchsorted(self.program.sets, ratio_0) - 1
            if self.next_layer_0 < 0:
                self.next_layer_0 = 0
            if r_test_1.size == 0:
                ratio_1 = 100
            else:
                ratio_1 = p_test_1.size / r_test_1.size
            self.next_layer_1 = np.searchsorted(self.program.sets, ratio_1) - 1
            if self.next_layer_1 < 0:
                self.next_layer_1 = 0

            self.program.nodes[self.level+1][self.next_layer_0].pts_train_p = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_0].pts_train_p, p_train_0))
            self.program.nodes[self.level + 1][self.next_layer_1].pts_train_p = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_1].pts_train_p, p_train_1))

            self.program.nodes[self.level + 1][self.next_layer_0].pts_train_r = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_0].pts_train_r, r_train_0))
            self.program.nodes[self.level + 1][self.next_layer_1].pts_train_r = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_1].pts_train_r, r_train_1))

            self.program.nodes[self.level + 1][self.next_layer_0].pts_test_p = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_0].pts_test_p, p_test_0))
            self.program.nodes[self.level + 1][self.next_layer_1].pts_test_p = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_1].pts_test_p, p_test_1))

            self.program.nodes[self.level + 1][self.next_layer_0].pts_test_r = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_0].pts_test_r, r_test_0))
            self.program.nodes[self.level + 1][self.next_layer_1].pts_test_r = np.vstack((
                self.program.nodes[self.level + 1][self.next_layer_1].pts_test_r, r_test_1))
        else:
            self.no_split()

    def no_split(self):
       # print("didn't split", self.level, self.height)
        self.program.nodes[self.level + 1][self.height].pts_train_p = \
            np.vstack((self.pts_train_p, self.program.nodes[self.level + 1][self.height].pts_train_p))
        self.program.nodes[self.level + 1][self.height].pts_train_r = \
            np.vstack((self.pts_train_r, self.program.nodes[self.level + 1][self.height].pts_train_r))
        self.program.nodes[self.level + 1][self.height].pts_test_p = \
            np.vstack((self.pts_test_p, self.program.nodes[self.level + 1][self.height].pts_test_p))
        self.program.nodes[self.level + 1][self.height].pts_test_r = \
            np.vstack((self.pts_test_r, self.program.nodes[self.level + 1][self.height].pts_test_r))

    # assigns labels, 0 for p, 1 for r. Calculates weights so that the total weight is equal
    @staticmethod
    def prepare_points_weigh(p_pts, r_pts):
        p_size = p_pts[:, 0].size
        r_size = r_pts[:, 0].size
        weights = np.ones(p_size)
        X_train = np.concatenate((p_pts, r_pts), axis=0)
        y_train = np.concatenate((np.zeros(p_size), np.ones(r_size)), axis=0)
        if r_size > 0 and p_size == 0:
            weights = np.ones(r_size)
        else:
            if r_size > 0 and p_size > 0:
                weights = np.concatenate((np.ones(p_size), (p_size / r_size) * np.ones(r_size)), axis=0)
        return X_train, y_train, weights

    # assigns labels, 0 for p, 1 for r. Downsamples whichever one has more data
    @staticmethod
    def prepare_points_sample(p_pts, r_pts):
        p_size = p_pts[:, 0].size
        r_size = r_pts[:, 0].size
        if p_size > r_size:
            min_size = r_size
            subset = np.random.choice(p_size, size=min_size, replace=True)
            p_pts = p_pts[subset, :]
        else:
            min_size = p_size
            subset = np.random.choice(r_size, size=min_size, replace=True)
            r_pts = r_pts[subset, :]
        X_train = np.concatenate((p_pts, r_pts), axis=0)
        y_train = np.concatenate((np.zeros(min_size), np.ones(min_size)), axis=0)
        return X_train, y_train

    def predict(self):

        # print("In predict, at level ", self.level, " and height", self.height)
        if self.pts_predicted.size == 0:
            return
        if not self.was_fitted:
            self.program.nodes[self.level + 1][self.height].pts_predicted = \
                np.vstack((self.program.nodes[self.level + 1][self.height].pts_predicted, self.pts_predicted))
            # print("was not fitted before predict!")
            return
        y = self.classifier.predict(self.pts_predicted)
        pts_0 = self.pts_predicted[y == 0]
        pts_1 = self.pts_predicted[y == 1]

        # print("In predict, at level ", self.level, " and height", self.height)
        # print("split [", len(pts_0), "] as 0")
        # print("split [", len(pts_1), "] as 1")
        self.program.nodes[self.level + 1][self.next_layer_0].pts_predicted = \
            np.vstack((self.program.nodes[self.level + 1][self.next_layer_0].pts_predicted, pts_0))
        self.program.nodes[self.level + 1][self.next_layer_1].pts_predicted = \
            np.vstack((self.program.nodes[self.level + 1][self.next_layer_1].pts_predicted, pts_1))
