import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from numpy import reshape
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class ModelBuilder:
    # Constructor with model type specification. These are the current supported models that I have found.
    # Feel free to add more. Each of these will be evaluated each cycle to determine best fit to dataset.
    def __init__(self, model_type):
        if model_type == 'LR':
            self.model = LogisticRegression()
        if model_type == 'KNN':
            self.model = KNeighborsClassifier(3)
        if model_type == 'CART':
            self.model = DecisionTreeClassifier(max_depth=5)
        if model_type == 'FORS':
            self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        if model_type == 'NB':
            self.model = GaussianNB()
        if model_type == 'GP':
            self.model = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
        if model_type == 'SVM Linear':
            self.model = SVC(kernel="linear", C=0.025)
        if model_type == 'SVM Gamma':
            self.model = SVC(gamma=2, C=1)
        if model_type == 'PAC':
            self.model = PassiveAggressiveClassifier()
        if model_type == 'ADA':
            self.model = AdaBoostClassifier(n_estimators=100)
        if model_type == 'LDA':
            self.model = LinearDiscriminantAnalysis()
        if model_type == 'QDA':
            self.model = QuadraticDiscriminantAnalysis()
        if model_type == 'MLP':
            self.model = MLPClassifier(alpha=1)
        if model_type == 'ETC':
            self.model = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                                   min_samples_split=2, random_state=0)

    # Apply the selected dataset to the model
    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    # Obtain result prediction from the model
    def get_predictions(self, x_valid):
        dims = np.ndim(x_valid)
        if dims == 1:
            # This is a one-dimensional array, which is deprecated. Fix now to avoid future problems.
            x_valid = self.format_single_sample(x_valid)
        return self.model.predict(x_valid)

    # Avoid angering the Sklearn library. Single dimension arrays will be deprecated in future versions
    # This method will turn it into a 2d array
    def format_single_sample(self, sample):
        # sample is an array of a single phrase's features
        temp = sample
        temp = np.array(temp).reshape((1, -1))
        return temp

    # Determine accuracy score given the validation set.
    def accuracy_score(self, x_valid, y_valid):
        return accuracy_score(y_valid, self.get_predictions(x_valid))

    # Print confusion matrix
    def confusion_matrix(self, x_valid, y_valid):
        print(confusion_matrix(y_valid, self.get_predictions(x_valid)))

    # Print classification report
    def classification_report(self, x_valid, y_valid):
        print(classification_report(y_valid, self.get_predictions(x_valid)))
