from pathlib import Path
import numpy as np
import skimage.io as ski
import skimage.feature as skif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

MODELFILE = "model.pkt"
MAX_ITER = 200   # Switch to 1600 to remove TOTAL. NO. of ITERATIONS REACHED LIMIT warning - still works fine not reaching convergence

"""
    Return:
        List of (foldername, [filenames...])
"""
def obtainFilenames():
    trainPath = Path("./train")
    ret = []
    for labelpath in trainPath.iterdir():
        ret.append((labelpath.name, [item for item in labelpath.iterdir() if item.is_file]))

    return ret

def getLabelSet(labelar):
    # Switch label names to label values
    labels = {}
    i = 0
    for label, _ in labelar:
        if labels.get(label) == None:
            labels[label] = i
            labels[i] = label
            i += 1

    return labels



def retrieveData():
    # Extract files from directories
    filenames = obtainFilenames()

    # Switch label names to label values
    labels = {}
    i = 0
    for label, _ in filenames:
        if labels.get(label) == None:
            labels[label] = i
            labels[i] = label
            i += 1

    # Separate train and test information
    trainFactor = 0.9
    trainlen = [int(len(lyst) * trainFactor) for _, lyst in filenames]
    trainFiles = [item for (_, lyst), l in zip(filenames, trainlen) for item in lyst[0:l]]
    trainlabels = np.reshape([labels[name] for (name, lyst), l in zip(filenames, trainlen) for item in lyst[0:l]], [len(trainFiles), 1])

    testFiles = [item for (_, lyst), l in zip(filenames, trainlen) for item in lyst[l:]]
    testlabels = [labels[name] for (name, lyst), l in zip(filenames, trainlen) for item in lyst[l:]]

    trainSize = len(trainFiles)
    testSize = len(testFiles)

    # Set up data
    X = np.zeros([trainSize, 30000])
    test = np.zeros([testSize, 30000])

    # Calculate training features
    print("Calculating training data features")
    for i in range(trainSize):
        filepath = trainFiles[i]

        if i % (trainSize // 10) == 0:
            print(f"Progress: {i} / {trainSize}")

        imgdata = np.asarray(ski.imread(filepath))
        X[i] = imgdata.flatten()

    # Calculate testing features
    print("\nCalculating testing data features")
    for i in range(testSize):
        filepath = testFiles[i]

        if i % (testSize // 10) == 0:
            print(f"Progress: {i} / {testSize}")

        imgdata = np.asarray(ski.imread(filepath))

        test[i] = imgdata.flatten()

    return X, trainlabels, test, testlabels, labels

def main(save, load):
    X, trainlabels, test, testlabels, labels = retrieveData()

    # Set up data
    LR:LogisticRegression = None
    ss:StandardScaler = None
    pca:PCA = None

    if load:
        import pickle
        print("Loading previous data")
        with open(MODELFILE, 'rb') as f:
            [LR, ss, pca, _] = pickle.load(f)
    else:
        # Scale data
        print("Fitting Standard Scalar")
        ss = StandardScaler()
        training_ss = ss.fit_transform(X, trainlabels)
        del X

        # Reduce data dimensionaliy
        print("Fitting PCA")
        pca = PCA(200)
        training_pca = pca.fit_transform(training_ss, trainlabels)
        del training_ss

        # Fit data with Logistic Regression
        print("Fitting Logistic Regression")
        LR = LogisticRegression(multi_class='ovr', max_iter=MAX_ITER)
        LR.fit(training_pca, trainlabels.flatten())

    print("Transforming Testing Data")
    testing_ss = ss.transform(test)
    del test
    testing_pca = pca.transform(testing_ss)
    del testing_ss

    # Save model for later
    if save:
        import pickle
        print(f"Saving model to {MODELFILE}")
        with open(MODELFILE, 'wb') as f:
            pickle.dump([LR, ss, pca, labels], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Test model against testing data
    correct = 0
    total = 0
    errors = []
    print("Testing model against testing data")
    for data, label in zip(testing_pca, testlabels):
        prediction = int(LR.predict(data.reshape(1,-1)))
        total += 1
        if prediction == label:
            correct += 1
        else:
            errors.append([labels[prediction], labels[label]])

    print(f"Accuracy: {correct} / {total} = {((correct / total) * 100):.2f}%")
    for [prediction, actual] in errors:
        print(f"{prediction} /= {actual}")

def against(file):
    # Load previous data
    LR:LogisticRegression = None
    ss:StandardScaler = None
    pca:PCA = None
    lables:dict = None

    import pickle
    print("Loading previous data")
    with open(MODELFILE, 'rb') as f:
        [LR, ss, pca, labels] = pickle.load(f)

    # Extract new data from file
    imgdata = np.asarray(ski.imread(file))
    test = imgdata.flatten()

    # Format new data
    test_ss = ss.transform(test.reshape(1, -1))
    test_pca = pca.transform(test_ss)

    # Test against new data
    prediction = int(LR.predict(test_pca.flatten().reshape(1, -1)))
    print(labels[prediction])

if __name__=="__main__":
    # Specific imports for this section
    import sys
    from parameterReader import ParameterReader

    # Set up ParameterReader
    defParams = {"--save":False, "-f":"", "--load":False}
    p = ParameterReader(defParams)

    # Read params
    try:
        p.setParams(sys.argv)
        if defParams["-f"] == "":
            # Setup new model (save if set to)
            main(defParams["--save"], defParams["--load"])
        else:
            # Run against a new image
            against(defParams["-f"])

    except NameError as e:
        print(f"Error reading parameters: {e.args[0]}")

    