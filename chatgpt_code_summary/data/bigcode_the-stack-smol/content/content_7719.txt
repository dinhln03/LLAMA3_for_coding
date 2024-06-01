# Import packages
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

# Import model builder from model_build.py
from model_build import DRbuild

# Load MNIST dataset
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Add grayscale channel dimension
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Scale data to [0, 1] range
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Enconde label to vector
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# starting learning rate
LR = 1e-3
# Epochs to train
EPOCHS = 10
# Batch size
BS = 128

# Compile model
opt = Adam(lr=LR)
model = DRbuild(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Train model
H = model.fit(
    trainData, trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BS,
    epochs=EPOCHS,
    verbose=1)

# Evaluate model
predictions = model.predict(testData)

# Serialize model
path = os.getcwd()
path = path[:path.rfind('\\') + 1]
path = path + r'models\digit_classifier.h5'
model.save(path, save_format="h5")
