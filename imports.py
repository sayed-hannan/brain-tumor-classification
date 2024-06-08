# Common imports for data processing
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, callbacks, Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.saving import register_keras_serializable


# Other common imports
from utility.data_utils import print_shapes, plot_images
from data_loading import DatasetLoader
from data_augmentation import DataAugmentation
from data_optimization import DatasetOptimizer
from model import CustomCNN
from base_model import  BaseModel 
# from vgg_16 import CustomVGG16
# from pre_trained import PreTrainedVGG16

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
