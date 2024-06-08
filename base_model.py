from imports import *

TF_ENABLE_ONEDNN_OPTS = 0

class BaseModel(Model):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.5)
        self.flatten = layers.Flatten()
        self.dropout2 = layers.Dropout(0.5)
        self.fc1 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        return self.fc1(x)
