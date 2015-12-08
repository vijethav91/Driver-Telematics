import classifiers
import csv
import os

def writeCsv(clf, outputWriter):
    for i in range(200):
        outputWriter.writerow([clf.ids[i], clf.label[i]])

# configurations to Run in Batch mode
classifiersList = ['LogisticRegression', 'RandomForest', 'GBM']
numTrips = [100, 200, 400, 800, 1600, 3200, 5000]
numComponents = [10, 30, 50, 70]
dimRedType = ['PCA', 'LDA', 'ICA']

# Factory method to instantiate the classifier
def classifierFactory(name, runDimRed, dimRedType, sampleType, numDrivers, numTrips):
    if name == 'LogisticRegression':
        return classifiers.SimpleLogisticRegression(runDimRed, dimRedType, sampleType, numDrivers, numTrips)
    if name == 'RandomForest':
        return classifiers.RandomForest(runDimRed, dimRedType, sampleType, numDrivers, numTrips)
    if name == 'GBM':
        return classifiers.GBM(runDimRed, dimRedType, sampleType, numDrivers, numTrips)

if __name__ == "__main__":
    # list all the driver feature files
    driverData = os.listdir(classifiers.Classifier.baseFeatureFolder)
    
    # measure the effect of trip sampling on classifier performance
    for clfName in classifiersList:
        # vanilla classifier
        clf = classifierFactory(clfName, False, '', 1, 1, 1)
        clf.loadAllFeatures(driverData)
        for numTrip in numTrips:
            print "Running RandomForest with %d trips" % (i)
            # set the number of trips
            clf.numTrips = numTrip

            outFile = open('T' + str(i) + '_' + clf.outputFileName, 'wb')
            outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
            outputWriter.writerow(['driver_trip', 'prob'])

            for driver in driverData:
                # sanity check to skip '.DS_Store' file in Mac
                if driver.startswith('.'):
                    continue
                #print "Processing driver ", driver

                clf.runClassifier(driver, 0)
                writeCsv(clf, outputWriter)
            outFile.close()
    
    # measure the performance of number of components from different dimensionality reduction methods
    for clfName in classifiersList:
        for dimRed in dimRedType:
            # vanilla classifier
            clf = classifierFactory(clfName, True, dimRed, 1, 1, 1600)
            clf.loadAllFeatures(driverData)

            for numComponent in numComponents:
                print "Running %s with %s and %d components" % (clfName, dimRed, numComponent)

                outFile = open('NC' + str(j) + '_' + clf.outputFileName, 'wb')
                outputWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
                outputWriter.writerow(['driver_trip', 'prob'])

                for driver in driverData:
                    # sanity check to skip '.DS_Store' file in Mac
                    if driver.startswith('.'):
                        continue
                    #print "Processing driver ", driver

                    clf.runClassifier(driver, numComponent)
                    writeCsv(clf, outputWriter)
                outFile.close()

