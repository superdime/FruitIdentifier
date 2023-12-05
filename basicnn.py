from pathlib import Path
import numpy as np
import skimage.io as ski
from skimage.color import rgb2gray
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


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
    testlabels = np.reshape([labels[name] for (name, lyst), l in zip(filenames, trainlen) for item in lyst[l:]], [len(testFiles), 1])

    trainSize = len(trainFiles)
    testSize = len(testFiles)

    # Set up data
    X = np.zeros([trainSize, 10000])
    test = np.zeros([testSize, 10000])

    # Calculate training features
    print("Calculating training data features")
    for i in range(trainSize):
        filepath = trainFiles[i]

        if i % (trainSize // 10) == 0:
            print(f"Progress: {i} / {trainSize}")

        imgdata = np.asarray(ski.imread(filepath))
        grayimgdata = rgb2gray(imgdata)
        X[i] = np.reshape(grayimgdata, (100*100))

    # Calculate testing features
    print("\nCalculating testing data features")
    for i in range(testSize):
        filepath = testFiles[i]

        if i % (testSize // 10) == 0:
            print(f"Progress: {i} / {testSize}")

        imgdata = np.asarray(ski.imread(filepath))
        grayimgdata = rgb2gray(imgdata)

        test[i] = np.reshape(grayimgdata, (100*100))

    return X, trainlabels, test, testlabels, labels

# Neural Network Model
class Model(nn.Module):
    def __init__(self, inputFeatures, hidden1Features, hidden2Features, outputFeatures):
        super().__init__()
        self.l1 = nn.Linear(inputFeatures, hidden1Features)
        self.l2 = nn.Linear(hidden1Features, hidden2Features)
        self.l3 = nn.Linear(hidden2Features, outputFeatures)

    def forward(self, x):
        x = self.l1(x)
        x = nn.Sigmoid()(x)
        x = self.l2(x)
        x = nn.Sigmoid()(x)
        x = self.l3(x)
        x = nn.Softmax(dim=1)(x)
        return x


def main(save, load):
    X, trainlabels, test, testlabels, labels = retrieveData()

    # Normalize and put into Tensor
    print("Normalizing training data")
    Xtrain = (X - np.mean(X)) / np.std(X)
    Xtrain = torch.from_numpy(Xtrain).float()
    ytrain = torch.from_numpy(trainlabels)

    # Set up Dataset
    trainDS = TensorDataset(Xtrain, ytrain)
    dataset = DataLoader(trainDS, len(X) // 10, shuffle=True)

    # Set up model
    print("Setting up model")
    model = Model(100*100, 1000, 100, 1)
    learningRate = 0.001
    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # Setup training sequence
    epochs = 100
    lossHist = [0] * epochs
    accuracyHist = [0] * epochs

    # Execute training
    print("Begin training")
    for epoch in range(epochs):
        # Printing progress
        if epoch % (epochs // 10) == 0:
            print(f"Training: {epoch}/{epochs}")
        
        for x, y in dataset:
            pred = model(x)
            loss = lossfn(pred, y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            lossHist[epoch] += loss.item()*y.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y).float()
            accuracyHist[epoch] += is_correct.sum()


    # Run against test data
    XTest = (test - np.mean(X)) / np.std(X)
    XTest = torch.from_numpy(XTest).float()
    yTest = torch.from_numpy(testlabels)
    predTest = model(XTest)

    correct = (torch.argmax(predTest, dim=1) == yTest).float()
    accuracy = correct.mean()
    
    print(f'Test Acc.: {accuracy:.4f}')


    """
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

        # Reduce data dimensionaliy
        print("Fitting PCA")
        pca = PCA(200)
        training_pca = pca.fit_transform(training_ss, trainlabels)

        # Fit data with Logistic Regression
        print("Fitting Logistic Regression")
        LR = LogisticRegression(multi_class='ovr', max_iter=MAX_ITER)
        LR.fit(training_pca, trainlabels.flatten())

    print("Transforming Testing Data")
    testing_ss = ss.transform(test)
    testing_pca = pca.transform(testing_ss)

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

    for [prediction, actual] in errors:
        print(f"{prediction} /= {actual}")

    """

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
    grayimgdata = rgb2gray(imgdata)
    test = np.reshape(grayimgdata, (100*100))

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