
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn.metrics import confusion_matrix, f1_score


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class TensorFlowNetwork_2Layer():
    def __init__(self, neuronsPerLayer, learningRate = 0.1):
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(512, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    def train(self, xVals, yVals, epochs = 10):
        self.model.fit(xVals, yVals, epochs = epochs)

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        # sig = __sigmoid(self, x)
        sig = x
        return sig * (1 - sig)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 10, minibatches = True, mbs = 100):
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            if minibatches:
                xBatches = self.__batchGenerator(xVals, mbs)
                yBatches = self.__batchGenerator(yVals, mbs)
            else:
                xBatches, yBatches = xVals, yVals
            for xBatch, yBatch in zip(xBatches, yBatches):
                x = np.asarray(xBatch).reshape(100,784)
                y = yBatch
                # Get layers
                layer1, layer2 = self.__forward(x)
                # Backprob
                layer2_error = layer2 - y
                layer2d = layer2_error * self.__sigmoidDerivative(layer2)
                layer1_error = np.dot(layer2d, self.W2.T)
                layer1d = layer1_error * self.__sigmoidDerivative(layer1)
                layer1a = np.dot(x.T, layer1d)
                layer2a = np.dot(layer1.T, layer2d)
                # Update weights
                self.W1 = self.W1 - layer1a * self.lr
                self.W2 = self.W2 - layer2a * self.lr


        

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

# Classifier for custom nn
def customClassifier(custom_nn, xTest):
    xTest = np.asarray(xTest).reshape(len(xTest),784)
    ans = []
    for entry in xTest:
        # Get prediction
        custom_pred = custom_nn.predict(entry)
        # Get max value
        custom_pred_max = np.NINF
        custom_pred_max_index = 0
        for index, val in enumerate(custom_pred):
            if val > custom_pred_max:
                custom_pred_max = val
                custom_pred_max_index = index
        # Get corresponding prediction and append
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[custom_pred_max_index] = 1
        ans.append(pred)
    return np.array(ans)

# Classifier for tensorflow nn
def tensorflowClassifier(tensorflow_nn, xTest):
    # Test = np.asarray(xTest).reshape(len(xTest),784)
    # ans = []
    # for entry in tensorflow_nn.model.predict(xTest):
    #     # Get max value
    #     tensorflow_pred_max = np.NINF
    #     tensorflow_pred_max_index = 0
    #     for index, val in enumerate(tensorflow_pred):
    #         if val > tensorflow_pred_max:
    #             tensorflow_pred_max = val
    #             tensorflow_pred_max_index = index
    #     # Get corresponding prediction and append
    #     pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     pred[tensorflow_pred_max_index] = 1
    #     ans.append(pred)
    # return np.array(ans)
    return None




#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain / 255
    xTest = xTest / 255
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        custom_nn = NeuralNetwork_2Layer(inputSize=784, outputSize=10, neuronsPerLayer=50, learningRate=0.1)
        custom_nn.train(xTrain, yTrain)
        return custom_nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        tensorflow_nn = TensorFlowNetwork_2Layer(neuronsPerLayer = 50, learningRate = 0.1)
        tensorflow_nn.train(xTrain, yTrain)
        return tensorflow_nn
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return customClassifier(model, data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        temp = to_categorical(np.argmax(model.model.predict(data), axis=1), NUM_CLASSES)
        return temp
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(yTest.argmax(axis = 1), preds.argmax(axis = 1)))
    print()
    print("F1 Score:")
    print(f1_score(yTest.argmax(axis = 1), preds.argmax(axis = 1), average = 'micro'))




#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
