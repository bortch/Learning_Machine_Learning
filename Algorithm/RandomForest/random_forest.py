import numpy as np
from ..DecisionTreeClassifier.decision_tree_classifier import DecisionTree_Classifier
# to run script using relative path,
# from Learning_Machine_Learning directory, call:
# python -m Algorithm.RandomForest.random_forest

class RandomForest():
    """docstring for RandomForest."""

    def __init__(self, n_estimators=100, max_samples=.8, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.trees = [None for i in range(n_estimators)]

    def fit(self, X, y):
        n_samples, _ = X.shape
        for i in range(self.n_estimators):
            tree = DecisionTree_Classifier(max_depth=self.max_depth,
                                           criterion=self.criterion,
                                           auto_encode=False)
            idx = self._get_idx(n_samples)
            self.trees[i] = tree.fit(X[idx, :], y[idx])
        #print(f"Trees in the Forest:\n {self.trees}")
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            y_pred[i] = self._aggregate(X[i, :].reshape(1, n_features))
        return y_pred

    def _aggregate(self, X):
        y_pred = np.empty(len(self.trees), dtype=np.int64)
        for i, tree in enumerate(self.trees):
            y_pred[i] = tree.predict(X)[0]
        class_, counts = np.unique(y_pred, return_counts=True)
        return class_[np.argmax(counts)].astype(int)

    def _get_idx(self, n_samples):
        return np.random.choice(n_samples, int(n_samples*self.max_samples), replace=False)

    def _get_resampled(self, X, y):
        n_samples, n_features = X.shape
        idx = self._get_idx(n_samples, n_features)
        return X[idx, :], y[idx]

    def _get_fitted_tree(self, tree, X, y):
        # t = {#"tree": tree.get_tree(),
        #      # as we don't want to store np.array
        #      "y_pred": tree.predict(X)#,
        #      #"score": tree.score(X_test, y_test)
        #      }
        return tree.fit(X, y)

    def score(self, X, y):
        y_pred = self.predict(X)
        good_pred = y_pred == y
        good_pred = good_pred.sum()
        good_pred = good_pred/y_pred.shape
        return good_pred[0]


if __name__ == "__main__":
    from time import perf_counter as pc

    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    np.random.seed(1)

    # preparing dataset
    data = load_iris(as_frame=True)['frame']
    X = data.drop(columns='target').values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6)

    rf = RandomForest(n_estimators=100, max_samples=.6)
    start = pc()
    rf.fit(X_train, y_train)
    print(pc()-start)

    y_pred = rf.predict(X_test)
    results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    print(f"Prediction:\t{y_pred}")
    print(f"y test target:\t{y_test}")
    score = rf.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Score:{score}\nAccuracy: {accuracy}\nF1 Score: {f1}")
