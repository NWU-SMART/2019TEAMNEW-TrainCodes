from keras.models import Sequential
from keras.layers.core import Reshape, Permute, Activation
from util import date_dimension_reduction

def creat_cate_model(name="cate1"):

    model.add(Dense(output_dim=80, input_dim=80, activation='relu'))
    model.add(Dense(output_dim=20, input_dim=80, activation='relu'))
    model.add(Dense(output_dim=5, input_dim=20, activation='relu'))
    model.add(Dense(output_dim=5, input_dim=5, activation='relu'))
    model.add(Dense(output_dim=80, input_dim=5, activation='relu'))
    return model
class CATE1(object):
    def __init__(self, label=None):
