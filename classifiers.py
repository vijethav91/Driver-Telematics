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
        self.outputFileName = self.clfName + "_Output.csv"

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
        print "Loading all features"
        for _driver in _driverDataFolder:
            if _driver.startswith('.'):
                continue
            self.globalFeatureHash[_driver] = super(SimpleLogisticRegression, self).loadFeatures(_driver)
        print "Done Loading all features"

    def randomSampler(self, driver, sampleType=1, numDrivers=1, numTrips=1):
        # create the positive trips
        _X = self.globalFeatureHash[driver].values()

        # create the negative trips based on sampleType
        if sampleType == 1:
            # based on number of random trips
            totalTrips = len(self.globalFeatureHash)*200
            randomInts = np.random.random_integers(0, totalTrips-1, numTrips)
            secondaryDriverKeys = map(lambda x:x/200, randomInts)
            secondaryDriverIDs = map(lambda x:self.globalFeatureHash.keys()[x], secondaryDriverKeys)
            secondaryTripKeys = map(lambda x:x%200, randomInts)
            randomSampleTrips = map(lambda i:self.globalFeatureHash[secondaryDriverIDs[i]].values()[secondaryTripKeys[i]] , range(numTrips))
        else:
            # based on number of random drivers
            secondaryFeatureKeys = np.random.choice(self.globalFeatureHash.keys(), numDrivers, replace=False)
            try:
                secondaryFeatureKeys.remove(driver)
                numDrivers = numDrivers - 1
            except:
                pass
            if numTrips == 1:
                randomSampleTrips = map(lambda x: self.globalFeatureHash[x].values()[np.random.choice(200, numTrips)], secondaryFeatureKeys)
            else:
                secondaryTripKeys = np.random.choice(200, numTrips)
                randomTrips = map(lambda x: map(lambda i: self.globalFeatureHash[x].values()[i], secondaryTripKeys), secondaryFeatureKeys)
                randomSampleTrips = reduce(lambda x,y: x+y, randomTrips)

        # add the negative trips to X
        _X.extend(randomSampleTrips)

        # construct the labels for the two classess accordingly
        _Y = np.append(np.ones(200), np.zeros(numDrivers*numTrips))

        return _X, _Y

    def runClassifier(self, _driverId, sampleType=1, numDrivers=1, numTrips=1):
        X, Y = self.randomSampler(_driverId, sampleType, numDrivers, numTrips)
        self.ids = self.globalFeatureHash[_driverId].keys()
        clf = LogisticRegression(class_weight='auto')
        model = clf.fit(X, Y)
        self.label = model.predict(X[:200])
