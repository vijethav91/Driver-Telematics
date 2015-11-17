import csv
import numpy as np
import os

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM as OCSVM

class Classifier(object):
    baseFeatureFolder = "features/"

    def __init__(self, clfName):
        self.clfName = clfName
        self.outputFileName = self.clfName+"_Output.csv"

    # Function to load the features from feature file
    def loadFeatures(self, _driverId):
        temp = {}
        self.label = []
        self.ids = []
        
        featureFileName = self.baseFeatureFolder + _driverId
        infile = open(featureFileName, 'r')
        infileReader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        infileReader.next()
        
        for line in infileReader:
            temp[line[0]] = np.nan_to_num(np.array(map(float, line[1:])))

        infile.close()

        return temp

    def getPCA(self, X):
        # normalize the features
        X_std = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=7)
        pca.fit(X_std)
        return pca.transform(X_std)

class OneClassSVM(Classifier):

    def __init__(self, nu, gamma, runpca=False):
        self.runpca = runpca
        clfName = 'OneClassSVM'
        if self.runpca:
            clfName = clfName + "_PCA"
        super(OneClassSVM, self).__init__(clfName)
        self.outliers_fraction = 0.03
        self.nu = nu
        self.gamma = gamma

    def loadFeatures(self, _driverId):
        self.featuresHash = super(OneClassSVM, self).loadFeatures(_driverId)

    def runClassifier(self):
        X = self.featuresHash.values()
        self.ids = self.featuresHash.keys()
        if self.runpca:
            X = self.getPCA(X)

        clf = OCSVM(nu=self.nu, gamma=self.gamma)
        clf.fit(X)
        y_pred = clf.decision_function(X).ravel()
        threshold = stats.scoreatpercentile(y_pred, 100 * self.outliers_fraction)
        self.label = y_pred > threshold
        self.label = map(int, self.label)

class SimpleLogisticRegression(Classifier):
    def __init__(self):
        super(SimpleLogisticRegression, self).__init__('SimpleLogisticRegression')
        self.globalFeatureHash = {}

    def loadAllFeatures(self, _driverDataFolder):
        #print "Loading all features"
        for _driver in _driverDataFolder:
            if _driver.startswith('.'):
                continue
            self.globalFeatureHash[_driver] = super(SimpleLogisticRegression, self).loadFeatures(_driver)
        #print "Done Loading all features"

    def samplerOne(self, num):
        secondaryFeatureKeys = np.random.choice(self.globalFeatureHash.keys(), num, replace=False)
        return reduce(lambda x,y: x+y, map(lambda x:self.globalFeatureHash[x].values(), secondaryFeatureKeys)))

    def samplerTwo(self, num):
        secondaryFeatureKeys = np.random.choice(self.globalFeatureHash.keys(), num, replace=False)
        return reduce(lambda x,y: x+y, map(lambda x:self.globalFeatureHash[x].values(), secondaryFeatureKeys)))        

    def runClassifier(self, _driverId, numDrivers):
        #print "Computing X,Y"
        X = self.globalFeatureHash[_driverId].values()
        self.ids = self.globalFeatureHash[_driverId].keys()
        try:
            secondaryFeatureKeys.remove(_driverId)
            numDrivers = numDrivers - 1
        except:
            pass
        X.extend(self.samplerOne())
        Y = np.append(np.ones(200), np.zeros(numDrivers*200))
        #print len(X), len(Y)
        #print "Done computing X,Y"

        clf = LogisticRegression()
        model = clf.fit(X, Y)
        self.labels = model.predict(X[:200])

