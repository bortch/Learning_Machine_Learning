import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

class DecisionTree_Classifier(BaseEstimator):
    """DecisionTree_Classifier - Learn by Doing
    
    Keyword arguments:
    bandwidth -- the windows size (default None)
    criterion -- 'gini' (default Gini)
    max_depth -- None
    auto_encode -- True/False (default True)
    """

    def __init__(self, criterion='gini', max_depth=None, auto_encode=True):
        self.criterion = criterion #TODO add entropy
        self.max_depth = max_depth
        self.depth = 0
        self.auto_encode = auto_encode
        self._encoder = None

    def fit(self, X, y):
        if self.auto_encode == True:
            self._encoder = LabelEncoder()
            y = self._encoder.fit_transform(y)
        self.tree = self._get_split(X, y)
        # print(self.tree)
        return self

    def predict(self, X, tree=None):
        if(tree is None):
            tree = self.tree
        else:
            self.tree = tree
        n_samples, n_features = X.shape
        y_pred = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            y_pred[i] = self._get_prediction(X[i, :].reshape(1, n_features))
        # re-encode label initially passed
        if self.auto_encode == True:
            y_pred = self._encoder.inverse_transform(y_pred)
        return y_pred

    def _get_prediction(self, X, tree=None):
        if(tree is None):
            tree = self.tree
        if X[0, tree['feature']] <= tree['condition']:
            return self._evaluate_sub_tree(X, tree['left'])
        else:
            return self._evaluate_sub_tree(X, tree['right'])

    def _evaluate_sub_tree(self, X, sub_tree):
        if(isinstance(sub_tree, np.int64)):
            return sub_tree
        elif isinstance(sub_tree, dict):
            return self._get_prediction(X, sub_tree)
        else:
            return self._get_prediction(X, sub_tree)

    def _freq(self, _node):
        _, p = np.unique(_node, return_counts=True)
        freq_ = p/_node.shape[0]
        return freq_

    def _compute_gini_indice(self, _node):
        freq_ = self._freq(_node)
        gini = 1-((freq_**2).sum())
        return gini

    def _get_gini_indice(self, left_, right_):
        population_size = left_.shape[0]+right_.shape[0]
        gini_left_weighted = (
            left_.shape[0]/population_size) * self._compute_gini_indice(left_)
        gini_right_weighted = (
            right_.shape[0]/population_size) * self._compute_gini_indice(right_)
        split_gini = gini_left_weighted+gini_right_weighted
        return split_gini

    def _get_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = np.inf

        for i in range(n_features):
            for condition in np.unique(X[:, i]):
                left, right = y[X[:, i] <= condition], y[X[:, i] > condition]
                gini = self._get_gini_indice(left, right)
                if gini < best_gini:
                    best_gini = gini
                    node = {'condition': condition,
                            'feature': i,
                            'gini': best_gini}

        left = (X[X[:, node['feature']] <= node["condition"]],
                y[X[:, node['feature']] <= node["condition"]])
        right = (X[X[:, node['feature']] > node["condition"]],
                 y[X[:, node['feature']] > node["condition"]])
        self.depth += 1
        node['left'] = self.check_sub_node(left)
        node['right'] = self.check_sub_node(right)
        node['depth'] = self.depth
        return node

    def check_sub_node(self, sub_node):
        current_depth = self.depth
        if(self._is_leaf(sub_node)):
            return np.unique(sub_node[1])[0]
        elif(self._is_empty(sub_node)):
            return None
        else:
            if self.depth == self.max_depth:  # or self.min_samplesleaf
                return self._get_leaf(sub_node[1])
            else:
                node = self._get_split(sub_node[0], sub_node[1])
                self.depth = current_depth
                return node

    def _get_leaf(self, value):
        class_, count = np.unique(value, return_counts=True)
        class_ = class_[count == np.max(count)]
        return np.random.choice(class_)

    def _is_leaf(self, X):
        res = (len(np.unique(X[1])) == 1)
        return res

    def _is_empty(self, X):
        res = (len(np.unique(X[1])) == 0)
        return res

    def score(self, X, y):
        y_pred = self.predict(X)
        good_pred = y_pred == y
        good_pred = good_pred.sum()
        good_pred = good_pred/y_pred.shape
        return good_pred[0]

    def __repr__(self):
        return str(self.tree)

    def _beautify(self, dictionary, prefix=[], s=''):
        string_ = s
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # add the name of the node ("left" or "right")
                # and prefix it with spaces or appropriate characters.
                string_ += "".join(prefix)+f"_ {str(key)}:"
                # if it is a right-hand child (since we are starting from the left),
                # we can delete the "pipe" character (|)
                # and replace it with a sufficient space.
                if key == 'right':
                    prefix[-1] = "    "
                # as value is a dictionary, we parse it recursively
                string_ = self._beautify(value, prefix, string_+"\n")
                # when all sub tree are parsed
                if len(prefix) > 0:
                    # we go back from one level
                    prefix.pop()
                    # if the last node was a right one
                    if key == 'right':
                        # we go back once more
                        prefix.pop()
            elif not key == "depth":
                # inside a dictionnary
                if key == 'condition':
                    prefix.append("   ")
                    string_ += "".join(prefix)+f"< {str(value)} > "
                elif key == 'feature':
                    string_ += f"[{str(key)}: {str(value)},"
                elif key == 'gini':
                    string_ += f" {str(key)}: {str(round(value,4))}]\n"
                    prefix.append("   |")
                else:
                    # when we found a leaf
                    string_ += "".join(prefix)+f"_ {str(key)}: {str(value)}\n"
                    if key == 'right' and len(prefix) > 0:
                        prefix.pop()
        return string_

    def __str__(self):
        return self._beautify(self.tree)

    def get_tree(self):
        return self.tree


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score
    from time import perf_counter as pc

    np.random.seed(1)
    # Load dataset
    data = load_iris(as_frame=True)['frame']
    X = data.drop(columns='target').values
    y = data['target'].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)  # , random_state=0
    
    start = pc()
    dt = DecisionTree_Classifier()  # max_depth=4
    dt.fit(X_train, y_train)
    print(f"Decision Tree fit duration:{pc()-start}")

    start = pc()
    dt_sk = DecisionTreeClassifier()
    dt_sk.fit(X_train, y_train)
    print(f"SKlearn Decision Tree fit duration:{pc()-start}")
    
    print(str(dt))
    #X_test = np.array([6]).reshape(-1, 1)

    #tree = {'condition': 1.9, 'feature': 2, 'gini': 0.3349794238683128, 'left': '0', 'right': {'condition': 1.6, 'feature': 3, 'gini': 0.11297801866907561, 'left': {'condition': 4.9, 'feature': 2, 'gini': 0.03999999999999998, 'left': '1', 'right': {'condition': 6.0, 'feature': 0, 'gini': 0.2, 'left': {'condition': 2.2, 'feature': 1, 'gini': 0.0, 'left': '2', 'right': '1', 'depth': 5}, 'right': '2', 'depth': 4}, 'depth': 3}, 'right': {'condition': 4.8, 'feature': 2, 'gini': 0.03658536585365854, 'left': {'condition': 3.0, 'feature': 1, 'gini': 0.0, 'left': '2', 'right': '1', 'depth': 4}, 'right': '2', 'depth': 3}, 'depth': 2}, 'depth': 1}
    #results = []
    # for i in range(X_test.shape[0]):
    #    results.append(dt.predict(X_test[i,:].reshape(1,X_test.shape[1])))

    score = dt.score(X_test, y_test)
    score_sk = dt_sk.score(X_test, y_test)
    print(f"Score\ndt:{score}\nsk:{score_sk}")
    
    y_pred = dt.predict(X_test)
    results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    print(f"Prediction:\t{y_pred}")
    print(f"y test target:\t{y_test}")
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Score:{score}\nAccuracy: {accuracy}\nF1 Score: {f1}")
