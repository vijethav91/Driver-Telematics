import csv
import numpy as np
import os

from sklearn.svm import OneClassSVM
from scipy import stats

class Classifier:
    baseFeatureFolder = "features/"
    outputFileName = "Output.csv"

    def __init__(self):
        self.featuresHash = {}
        self.outliers_fraction = 0.03
        self.label = []
        self.ids = []

    # Function to load the features from feature file
    def loadFeatures(self, _driverId):
        featureFileName = self.baseFeatureFolder + _driverId
        infile = open(featureFileName, 'r')
        infileReader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        infileReader.next()
        for line in infileReader:
            self.featuresHash[line[0]] = np.nan_to_num(np.array(map(float, line[1:])))

    def runSVMClassifier(self):
        X = self.featuresHash.values()
        self.ids = self.featuresHash.keys()
        clf = OneClassSVM(nu=0.261, gamma=0.05)
        clf.fit(X)
        y_pred = clf.decision_function(X).ravel()
        threshold = stats.scoreatpercentile(y_pred, 100 * self.outliers_fraction)
        self.label = y_pred > threshold
        self.label = map(int, self.label)

    def writeCsv(self, writer):
        for i in range(200):
            outputWriter.writerow([self.ids[i], self.label[i]])


if __name__ == "__main__":
    driverData = os.listdir(Classifier.baseFeatureFolder)
    outFile = open(Classifier.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])
    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        testClf = Classifier()
        testClf.loadFeatures(driver)
        testClf.runSVMClassifier()
        testClf.writeCsv(outputWriter)
    outFile.close()
    # print len(testClf.featuresHash)
