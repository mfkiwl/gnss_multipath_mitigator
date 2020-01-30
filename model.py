from keras import layers
from keras import models
from keras import Input

class DopplerClassifier():
    def __init__(self, shape):
        # Create baseline
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
       	self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(4, activation='softmax'))

class DopplerRegressor:
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1))

class MultiTargetRegressor():
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(4))

class MiltiOutputRegressor():
    def __init__(self, shape):
        img_input = Input(shape=shape, dtype='float32', name='corr_img')
        x = layers.Conv2D(16, (3,3), activation='relu')(img_input)
        x = layers.Conv2D(16, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (3,3), activation='relu')(x)
        x = layers.Conv2D(32, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        
        x1 = layers.Dense(256, activation='relu')(x)
        x2 = layers.Dense(256, activation='relu')(x)
        
        dopp_pred = layers.Dense(1, name='doppler')(x1)
        delay_pred = layers.Dense(1, name='delay')(x2)
        
        self.model = models.Model(img_input, [dopp_pred, delay_pred])
        

class Model():
    def __init__(self, shape):
        # Create baseline
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
       	self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))


class Model10():
    def __init__(self, shape):
        # Create baseline
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
 
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

class Model8():
    def __init__(self, shape):
        # Create baseline
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

class Model4():
    def __init__(self, shape):
        # Create baseline
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
