import keras, math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.metrics import *
from keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import tensorflow_probability as tfp

# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# sess = tf.Session(config=tf.config)
# K.clear_session()
# K.set_session(sess)

def balanced_cross_entropy(y_true, y_pred, pos_weight=0.2):
    return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def R_squared(y, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

def PCC(y, y_pred):
    pcc = tfp.stats.correlation(y, y_pred, sample_axis=0, event_axis=None)
    return pcc

def mean_squared_logarithmic_error(y_true, y_pred):    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)    
    return K.mean(K.square(first_log - second_log), axis=-1)


def root_mean_squared_error(y_true, y_pred):
    '''
    Mean Absolute Error (MAE) is the average absolute difference between the true and predicted values.
    '''
    return tf.sqrt(tf.keras.losses.MeanSquaredError()(y_true, y_pred))


def MLPRegressor(input_dim, output_dims, loss=mean_squared_error, metrics=[mean_absolute_error, root_mean_squared_error, PCC]):
    model = Sequential()
    model.add(Dense(int(input_dim/2), input_shape=(input_dim,), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(int(input_dim/4), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(int(input_dim/6), activation='tanh'))
    model.add(Dense(int(input_dim/8), activation='tanh'))
    model.add(Dense(output_dims, activation='tanh'))
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model







