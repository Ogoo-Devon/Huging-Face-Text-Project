#Set Up and Installation
import tensorflow as tf
print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization, MaxPool1D


# Data Preprocessing
# from google.colab import files
uploaded = files.upload()

raw_data = pd.read_csv("output.csv")
data = raw_data.copy()
data.head()

data = data.drop(["Number"], axis=1, inplace=True)

X = data.drop(["Anaemic"], axis = 1)
y = data["Anaemic"]

le = LabelEncoder()
X["Sex"] = le.fit_transform(X["Sex"])
y_encode = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encode, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = X_train.reshape(-1, 6, 1)
X_test = X_test.reshape(-1, 6, 1)




# Building Model
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=(6, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv1D(filters=128, kernel_size=2, padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv1D(filters=256, kernel_size=2, padding = "valid", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.summary()






# Training Model
opt = tf.keras.optimizers.Adam(learning_rate=0.002)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])





# Model Evaluation and Prediction
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print(y_pred[1]), print(y_test[1])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

print(f"Accuracy Score:{accuracy_score(y_test, y_pred) * 100}%")





# plotting The Learning Curve
def learning_curve (history, epoch):
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history["accuracy"], label = "Accuracy")
  plt.plot(epoch_range, history.history["val_accuracy"], label = "Validation Accuracy")

  plt.plot(epoch_range, history.history["loss"], label = "Loss")
  plt.plot(epoch_range, history.history["val_loss"], label = "Validation Loss")
  plt.title("Learning Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Score")
  plt.legend()
  plt.show()

learning_curve(history, 200)









































































