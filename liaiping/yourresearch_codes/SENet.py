import keras
class SeNetBlock(object):
    def __init__(self,reduction=4):
        self.reduction = reduction
 
    def senet(self,input):
        channels = input.shape.as_list()[-1]
        avg_x = GlobalAveragePooling2D()(input)
        avg_x = Reshape((1,1,channels))(avg_x)
        avg_x = Conv2D(int(channels)//self.reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(avg_x)
        avg_x = Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(avg_x)
 
        max_x = GlobalMaxPooling2D()(input)
        max_x = Reshape((1,1,channels))(max_x)
        max_x = Conv2D(int(channels)//self.reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(max_x)
        max_x = Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(max_x)
 
        cbam_feature = Add()([avg_x,max_x])
 
        cbam_feature = Activation('hard_sigmoid')(cbam_feature)
 
        return Multiply()([input,cbam_feature])

