import classifiers
import csv
import numpy as np
import os

def writeCsv(clf, outputWriter):
    for i in range(200):
        outputWriter.writerow([clf.ids[i], clf.label[i]])

if __name__ == "__main__":
    driverData = os.listdir(classifiers.Classifier.baseFeatureFolder)

    # Running Logistic Regression with PCA
    slrClf = classifiers.SimpleLogisticRegression(runpca=True)
    slrClf.loadAllFeatures(driverData)

    outFile = open(slrClf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        slrClf.runClassifier(driver, 1, 1, 1600, 50)
        writeCsv(slrClf, outputWriter)
    outFile.close()
    exit()

    # Running Logistic Regression without PCA
    slrClf = classifiers.SimpleLogisticRegression()
    slrClf.loadAllFeatures(driverData)
    
    outFile = open(slrClf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        slrClf.runClassifier(driver, 1, 1, 1600)
        writeCsv(slrClf, outputWriter)
    outFile.close()
    exit()

    # Running One Class SVM without PCA
    svmClf = classifiers.OneClassSVM(0.261, 0.05)
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
    svmClf = classifiers.OneClassSVM(0.261, 0.05, True)
    outFile = open(svmClf.outputFileName, 'wb')
    outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    outputWriter.writerow(['driver_trip', 'prob'])

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        print "Processing driver ", driver
        svmClf.loadFeatures(driver)
        svmClf.runClassifier(7)
        writeCsv(svmClf, outputWriter)
    outFile.close()
