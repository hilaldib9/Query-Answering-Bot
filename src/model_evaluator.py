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


class ModelEvaluator:
    # Constructor
    # Parameters:
    #   - headers: Header array as provided by feature extractor's 'get_header_array' method
    #   - extracted_data: data matrix as provided by feature extractor's 'extract_word_features_dataset' method
    #   - seed: seed for random number generator
    #   - scoring: method for model evaluation. Options include 'accuracy' and some other things I don't remember now
    #   - validation_size: what percentage of the dataset should be set aside for testing purposes
    #   - debug: enable or disable debug output
    def __init__(self, headers, extracted_data, seed=7, scoring='accuracy', validation_size=0.2, debug=False):
        self.headers = headers
        self.extractedLines = extracted_data
        self.features = len(headers) - 1

        tmpfile = open('/tmp/out.csv', 'w')
        for item in extracted_data:
            tmpfile.write("%s\n" % item)

        self.seed = seed
        self.scoring = scoring
        self.validation_size = validation_size
        self.debug = debug

    # Display a graph of the features in relation to each other. Buggy on our current dataset.
    def view_matrix(self):
        scatter_matrix(self.dataset)
        plt.show()

    # See distribution of classifications
    def get_class_distribution(self):
        print(self.dataset.groupby('Classification').size())

    # Chop the dataset into train set and test set using params from constructor
    def split_dataset(self):
        dataset = pandas.read_csv("/tmp/out.csv", names=self.headers)
        array = dataset.values
        X = array[:, 0:self.features]
        Y = array[:, self.features]
        return model_selection.train_test_split(X, Y, test_size=self.validation_size, random_state=self.seed)

    # Find the algorithm that would be best for this dataset
    # Parameters:
    #   - points_noncollinear: set to true if the dataset contains data which are NOT either 0 or 1.
    #   - risk_convergence: set to true if you wish to include models which may not converge (may not work)
    def search_initial_best_fit_algorithm(self, points_noncollinear=False, risk_convergence=False):
        X_train, X_validation, Y_train, Y_validation = self.split_dataset()

        # #Spot Check Algorithms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier(3)))
        models.append(('CART', DecisionTreeClassifier(max_depth=5)))
        models.append(("FORS", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
        models.append(('NB', GaussianNB()))
        models.append(('GP', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)))
        models.append(('SVM Linear', SVC(kernel="linear", C=0.025)))
        models.append(('SVM Gamma', SVC(gamma=2, C=1)))
        models.append(('PAC', PassiveAggressiveClassifier()))
        models.append(('ADA', AdaBoostClassifier(n_estimators=100)))
        models.append(('ETC', ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=4,
                                                   random_state=0)))

        if points_noncollinear:
            models.add(("LDA", LinearDiscriminantAnalysis()))
            models.add(("QDA", QuadraticDiscriminantAnalysis()))
        if risk_convergence:
            models.append(('MLP', MLPClassifier(alpha=1)))

        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=7, random_state=self.seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=self.scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            if self.debug:
                print(msg)

        highest = 0
        for index in range(0, len(results) - 1):
            value = results[index].mean() - results[index].std()
            if value > highest:
                highest = value

        toReturn = []
        for index in range(0, len(results) - 1):
            value = results[index].mean() - results[index].std()
            if value == highest:
                toReturn.append(names[index])

        return toReturn, highest
