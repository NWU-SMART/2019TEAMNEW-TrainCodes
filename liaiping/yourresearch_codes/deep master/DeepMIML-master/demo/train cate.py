from __future__ import print_function
from keras.models import Sequential
from cocodemo import COCODataset, COCODataLayer
from deepmiml image refect import CATE
from deepmiml label refect import CATE1
from deepmiml.utils import save_keras_model
from cocodemo.vgg_16 import VGG_16
from util import date_dimension_reduction
if __name__ == "__main__":
    loss = "binary_crossentropy"
    nb_epoch = 10
    batch_size = 32
    model_name = "miml_vgg_16"

    # crate data layer
    dataset = COCODataset("data/coco", "train", "2014")
    data_layer = COCODataLayer(dataset, batch_size=batch_size)

    vgg_model_path = "models/imagenet/vgg/vgg16_weights.h5"
    base_model = VGG_16(vgg_model_path)
    fe = CATE(base_model=base_model)
    fe=fe.add(MaxPooling2D((1, n_instances), strides=(1, 1)))
    fe=fe.add(Dense(output_dim=20, input_dim=80, activation='relu'))
    base_model=base_model.add(MaxPooling2D((1, n_instances), strides=(1, 1)))
    base_model = base_model.add(Dense(output_dim=20, input_dim=80, activation='relu'))
    FE=fe+base_model
    print("Compiling cate Model...")
    deepmiml.model.compile(optimizer="adadelta", loss=loss, metrics=["accuracy"])

    print("Start Training...")
    samples_per_epoch = data_layer.num_images
    fe.model.fit_generator(data_layer.generate(),
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch)

    save_keras_model1(fe.model, "outputs/{}/{}".format(dataset.name, model_name))