from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import wandb
import mne 

def identity_block(X, f, filters, stage, block, l1=0.0, l2=0.01):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), data_format='channels_first',
               padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), data_format='channels_first', padding = 'same',
               name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), data_format='channels_first', padding = 'valid',
               name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
  
def convolutional_block(X, f, filters, stage, block, s=2, l1=0.0, l2=0.01):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s),
               data_format='channels_first', padding='same',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               data_format='channels_first', padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               data_format='channels_first', padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), data_format='channels_first', 
                        padding='same', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(X_shortcut)
    X_shortcut = BatchNormalization(axis=1, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet(chans=64, samples=121, l1=0.0, l2=0.01):
  input_shape= (1, chans, samples)
  input1   = Input(shape = input_shape)

  X = Conv2D(128, (1, 64), padding = 'same',
                        input_shape = input_shape, data_format='channels_first',
                        kernel_initializer = glorot_uniform(seed=0),
              activity_regularizer=l1_l2(l1, l2))(input1)
  X = BatchNormalization(axis = 1)(X)
  X = DepthwiseConv2D((64, 1), kernel_initializer = glorot_uniform(seed=0), 
                                 depth_multiplier = 2, data_format='channels_first',
              activity_regularizer=l1_l2(l1, l2))(X)
  X = BatchNormalization(axis = 1)(X)
  X = Activation('elu')(X)
  X = AveragePooling2D((1, 2), data_format='channels_first')(X)

  X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1, l1=l1, l2=l2)
  X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', l1=l1, l2=l2)
  X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', l1=l1, l2=l2)

  X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=1, l1=l1, l2=l2)
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='b', l1=l1, l2=l2)
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='c', l1=l1, l2=l2)
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='d', l1=l1, l2=l2)


  X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

  X = Flatten()(X)
  X = Dense(64, activation='relu')(X)
  X = Dense(32, activation='relu')(X)
  X = Dense(1, activation='sigmoid', name='fc1', kernel_initializer = glorot_uniform(seed=0))(X)


  model = Model(inputs = input1, outputs = X, name='ResNet50')

  return model

def inception_resnet(nb_classes=3, Chans = 64, Samples = 321, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', gpu=True):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))

    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False, data_format='channels_first')(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.), data_format='channels_first')(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 2), data_format='channels_first')(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same', data_format='channels_first')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 4), data_format='channels_first')(block2)
    block2       = dropoutType(dropoutRate)(block2)
    
    block3       = Conv2D(F2, (1, 8), padding='same', data_format='channels_first')(block2)
    block3       = BatchNormalization(axis=1)(block3)
    block3       = Activation('elu')(block3)
    block3       = AveragePooling2D((1, 8), data_format='channels_first')(block3)
        
    flatten      = Flatten(name = 'flatten')(block3)
    
    dense        = Dense(1, name = 'out', kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def load_dataset(subjects, runs):
  
  # check if dataset is downloaded
  download_dataset(subjects, runs)
  
  raw = None
  
  for i in subjects[1:]:
      for f in eegbci.load_data(i, runs):
          
          if raw is None:
            raw = read_raw_edf(f, preload=True)
          else:
            try:
              raw = concatenate_raws([raw, read_raw_edf(f, preload=True)])
            except:
              print('subject {} failed to concatinate'.format(i))
          
  return raw
  
def preprocess(raw, event_id, use_filter = True, low_freq=7, high_freq=30, tmin=1, tmax=2):
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    if use_filter:
      raw.filter(low_freq, high_freq)


    events, _ = get_events(raw)

    picks = get_picks(raw)

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = get_epochs(raw, events, event_id)

    epochs_train = epochs.copy().crop(tmin=tmin, tmax=tmax)
    labels = epochs_train.events[:, -1] - 2

    labels = labels.reshape((labels.shape[0],1))
    epochs_data_train = epochs_train.get_data()
    epochs_data_train = epochs_data_train.reshape((epochs_data_train.shape[0],1, epochs_data_train.shape[1], epochs_data_train.shape[2]))

    return epochs_data_train, labels


def get_events(raw, event_id=dict(T1=2, T2=8)):
    return events_from_annotations(raw, event_id=event_id)

def get_epochs(raw,events, event_id):
  
    return Epochs(raw, events, event_id, -1, 4, proj=True, picks=get_picks(raw),
                  baseline=(None, None), preload=True)
  
  
def get_picks(raw):
    return pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

model = Resnet()
model2 = inception_resnet()

sweep_config = {
    'method': 'bayesian', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [2, 5, 10]
        },
        'batch_size': {
            'values': [128, 64, 32]
        },
        'conv_layer_size': {
            'values': [16, 32, 64]
        },
        'weight_decay': {
            'values': [0.0005, 0.005, 0.05]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        }
    }
}

optimizer = optimizers.Adam(lr=0.001, decay=1e-5)
model.compile(optimizer , loss=losses.binary_crossentropy , metrics=['acc'])

batch_size = 256

def train(): 
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=25, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=30, validation_data=(X_test, y_test),
                    callbacks=[ mcp_save, reduce_lr_loss])
    wandb.log({"history": history, "epoch": epoch})
    model.evaluate(X_train, y_train)


wandb.agent(sweep_id, train, count=15)

