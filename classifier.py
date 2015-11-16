import csv
import numpy as np
import os

from sklearn.svm import OneClassSVM as OCSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

def writeCsv(clf, outputWriter):
    for i in range(200):
        outputWriter.writerow([clf.ids[i], clf.label[i]])

class Classifier(object):
    baseFeatureFolder = "features/"

    def __init__(self, clfName):
        self.clfName = clfName
        self.outputFileName = self.clfName+"_Output.csv"

    # Function to load the features from feature file
    def loadFeatures(self, _driverId):
        self.featuresHash = {}
        self.label = []
        self.ids = []
        featureFileName = self.baseFeatureFolder + _driverId
        infile = open(featureFileName, 'r')
        infileReader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        infileReader.next()
        for line in infileReader:
            self.featuresHash[line[0]] = np.nan_to_num(np.array(map(float, line[1:])))

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

if __name__ == "__main__":
    driverData = os.listdir(Classifier.baseFeatureFolder)

    # Running One Class SVM without PCA
    svmClf = OneClassSVM(0.261, 0.05)
    outFile = open(svmClf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        svmClf.loadFeatures(driver)
        svmClf.runClassifier()
        writeCsv(svmClf, outputWriter)
    outFile.close()

    # Running One Class SVM with PCA
    svmClf = OneClassSVM(0.261, 0.05, True)
    outFile = open(svmClf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        svmClf.loadFeatures(driver)
        svmClf.runClassifier()
        writeCsv(svmClf, outputWriter)
    outFile.close()
