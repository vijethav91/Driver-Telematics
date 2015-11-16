import classifiers
import csv
import numpy as np
import os

def writeCsv(clf, outputWriter):
    for i in range(200):
        outputWriter.writerow([clf.ids[i], clf.label[i]])

if __name__ == "__main__":
    driverData = os.listdir(classifiers.Classifier.baseFeatureFolder)

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
        svmClf.runClassifier()
        writeCsv(svmClf, outputWriter)
    outFile.close()
