import argparse
import classifiers
import csv
import numpy as np
import os

def writeCsv(clf, outputWriter):
    for i in range(200):
        outputWriter.writerow([clf.ids[i], clf.label[i]])

def classifierFactory(name, runPCA, sampleType, numDrivers, numTrips):
    if name == 'OneClassSVM':
        return classifiers.OneClassSVM(0.261, 0.05, runPCA)
    if name == 'LogisticRegression':
        return classifiers.SimpleLogisticRegression(runPCA, sampleType, numDrivers, numTrips)
    if name == 'RandomForest':
        return classifiers.RandomForest(runPCA, sampleType, numDrivers, numTrips)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", help="specify the ML algorithm to be used",
                        choices=['OneClassSVM', 'LogisticRegression', 'RandomForest'])
    parser.add_argument("-rp", "--runpca", help="choose to enable PCA ", action="store_true")
    parser.add_argument("-s", "--sampleType", help="choose the sampling method to use", type=int, choices=[0, 1, 2])
    parser.add_argument("-d", "--numDrivers", help="indicate the number of drivers", type=int, default=1)
    parser.add_argument("-t", "--numTrips", help="indicate the number of trips", type=int, default=1)
    parser.add_argument("-nc", "--numComponents", help="indicate the number of components for PCA", type=int, default=0)

    return parser

if __name__ == "__main__":
    # Parse the command line args
    cmdParser = parseArgs()
    args = cmdParser.parse_args()

    driverData = os.listdir(classifiers.Classifier.baseFeatureFolder)

    # call the factory method to instantiate the appropriate classifier
    clf = classifierFactory(args.clf, args.runpca, args.sampleType, args.numDrivers, args.numTrips)

    # Running Logistic Regression with PCA
    if args.clf != 'OneClassSVM':
        clf.loadAllFeatures(driverData)

    outFile = open(clf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver

        if args.clf == 'OneClassSVM':
            clf.loadFeatures(driver)
        clf.runClassifier(driver, args.numComponents)
        writeCsv(clf, outputWriter)
    outFile.close()
