import csv
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from scipy import stats
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM as OCSVM

class Classifier(object):
    baseFeatureFolder = "features/"

    def __init__(self, clfName, dimRedType='', sampleType=1, numDrivers=1, numTrips=1):
        self.globalFeatureHash = {}
        self.sampleType = sampleType
        self.numDrivers = numDrivers
        self.numTrips = numTrips
        self.clfName = clfName 
        self.dimRedType = dimRedType
        self.outputFileName = self.clfName + self. dimRedType + "_Output.csv"

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

    def loadAllFeatures(self, _driverDataFolder):
        print "Loading all features"
        for _driver in _driverDataFolder:
            if _driver.startswith('.'):
                continue
            self.globalFeatureHash[_driver] = self.loadFeatures(_driver)
        print "Done Loading all features"

    def randomSampler(self, driver):
        # create the positive trips
        _X = self.globalFeatureHash[driver].values()

        # create the negative trips based on sampleType
        if self.sampleType == 1:
            # based on number of random trips
            totalTrips = len(self.globalFeatureHash)*200
            randomInts = np.random.random_integers(0, totalTrips-1, self.numTrips)
            secondaryDriverKeys = map(lambda x:x/200, randomInts)
            secondaryDriverIDs = map(lambda x:self.globalFeatureHash.keys()[x], secondaryDriverKeys)
            secondaryTripKeys = map(lambda x:x%200, randomInts)
            randomSampleTrips = map(lambda i:self.globalFeatureHash[secondaryDriverIDs[i]].values()[secondaryTripKeys[i]] , range(self.numTrips))
        else:
            # based on number of random drivers
            secondaryFeatureKeys = np.random.choice(self.globalFeatureHash.keys(), self.numDrivers, replace=False)
            try:
                secondaryFeatureKeys.remove(driver)
                self.numDrivers = self.numDrivers - 1
            except:
                pass
            if self.numTrips == 1:
                randomSampleTrips = map(lambda x: self.globalFeatureHash[x].values()[np.random.choice(200, self.numTrips)], secondaryFeatureKeys)
            else:
                secondaryTripKeys = np.random.choice(200, self.numTrips)
                randomTrips = map(lambda x: map(lambda i: self.globalFeatureHash[x].values()[i], secondaryTripKeys), secondaryFeatureKeys)
                randomSampleTrips = reduce(lambda x,y: x+y, randomTrips)

        # add the negative trips to X
        _X.extend(randomSampleTrips)

        # construct the labels for the two classess accordingly
        _Y = np.append(np.ones(200), np.zeros(self.numDrivers*self.numTrips))

        return _X, _Y

    def getPCA(self, X, numComponents):
        # normalize the features
        X_std = StandardScaler().fit_transform(X)

        # Principal component analysis (PCA)
        pca = PCA(n_components=numComponents)
        X_pca = pca.fit_transform(X_std)
        
        return X_pca

    def getNMF(self, X, numComponents):
        # Non-Negative Matrix Factorization (NMF)
        nmf = NMF(n_components=numComponents, init='random', random_state=0)
        X_nmf = nmf.fit_transform(X)

        return X_nmf

    def getFastICA(self, X, numComponents):
        # Independent Component Analysis (ICA)
        ica = FastICA(n_components=numComponents)
        X_ica = ica.fit_transform(X)

        return X_ica

    def getLDA(self, X, Y, numComponents):
        # Linear Discriminant Analysis (LDA)
        lda = LDA(n_components=numComponents)
        X_lda = lda.fit_transform(X, Y)

        return X_lda

    def dimRed(self, X, Y, numComponents):
        if self.dimRedType == 'PCA':
            return self.getPCA(X, numComponents)
        elif self.dimRedType == 'LDA':
            return self.getLDA(X, Y, numComponents)
        elif self.dimRedType == 'NMF':
            return self.getNMF(X, numComponents)
        elif self.dimRedType == 'ICA':
            return self.getFastICA(X, numComponents)

class OneClassSVM(Classifier):

    def __init__(self, nu, gamma, runDimRed=False, dimRedType=''):
        self.runDimRed = runDimRed
        super(OneClassSVM, self).__init__('OneClassSVM', dimRedType)
        self.outliers_fraction = 0.03
        self.nu = nu
        self.gamma = gamma

    def loadFeatures(self, _driverId):
        self.featuresHash = super(OneClassSVM, self).loadFeatures(_driverId)

    def runClassifier(self, _driverId, numComponents=0):
        X = self.featuresHash.values()
        self.ids = self.featuresHash.keys()
        if self.runDimRed:
            X = self.dimRed(X, numComponents)

        clf = OCSVM(nu=self.nu, gamma=self.gamma)
        clf.fit(X)
        y_pred = clf.decision_function(X).ravel()
        threshold = stats.scoreatpercentile(y_pred, 100 * self.outliers_fraction)
        self.label = y_pred > threshold
        self.label = map(int, self.label)

class SimpleLogisticRegression(Classifier):
    def __init__(self, runDimRed, dimRedType='', sampleType=1, numDrivers=1, numTrips=1):
        self.runDimRed = runDimRed
        super(SimpleLogisticRegression, self).__init__('SimpleLogisticRegression', dimRedType, sampleType, numDrivers, numTrips)

    def runClassifier(self, _driverId, numComponents=0):
        X, Y = self.randomSampler(_driverId)
        if self.runDimRed:
            X = self.dimRed(X, Y, numComponents)
        self.ids = self.globalFeatureHash[_driverId].keys()
        clf = LogisticRegression(class_weight='balanced')
        model = clf.fit(X, Y)
        self.label = model.predict_proba(X[:200]).T[1]

class RandomForest(Classifier):
    def __init__(self, runDimRed, dimRedType='', sampleType=1, numDrivers=1, numTrips=1):
        self.runDimRed = runDimRed
        super(RandomForest, self).__init__('RandomForest', dimRedType, sampleType, numDrivers, numTrips)

    def runClassifier(self, _driverId, numComponents=0):
        X, Y = self.randomSampler(_driverId)
        if self.runDimRed:
            X = self.dimRed(X, Y, numComponents)
        self.ids = self.globalFeatureHash[_driverId].keys()
        clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=15, min_samples_leaf=5, n_jobs=-1)

        model = clf.fit(X, Y)
        self.label = model.predict_proba(X[:200]).T[1]

class GBM(Classifier):
    def __init__(self, runDimRed, dimRedType='', sampleType=1, numDrivers=1, numTrips=1):
        self.runDimRed = runDimRed
        super(GBM, self).__init__('GBM', dimRedType, sampleType, numDrivers, numTrips)

    def runClassifier(self, _driverId, numComponents=0):
        X, Y = self.randomSampler(_driverId)
        if self.runDimRed:
            X = self.dimRed(X, Y, numComponents)
        self.ids = self.globalFeatureHash[_driverId].keys()
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=15, min_samples_leaf=5)

        model = clf.fit(X, Y)
        self.label = model.predict_proba(X[:200]).T[1]

class MLP(Classifier):
    def __init__(self, runDimRed, dimRedType='', sampleType=1, numDrivers=1, numTrips=1):
        self.runDimRed = runDimRed
        super(MLP, self).__init__('MultilayerPerceptron', dimRedType, sampleType, numDrivers, numTrips)
        self.initmodel()
        self.sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    def initmodel(self):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=50, init='uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, init='uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, init='uniform'))
        self.model.add(Activation('softmax'))

    def runClassifier(self, _driverId, numComponents=0):
        X, Y = self.randomSampler(_driverId)
        if self.runDimRed:
            X = self.dimRed(X, Y, numComponents)

        self.ids = self.globalFeatureHash[_driverId].keys()

        self.model.compile(loss='mean_squared_error', optimizer=self.sgd)
        self.model.fit(X, Y, nb_epoch=500)

        self.label = self.model.predict_proba(X).T[0]
