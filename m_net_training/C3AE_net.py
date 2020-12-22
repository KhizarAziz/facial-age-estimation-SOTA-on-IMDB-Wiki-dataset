from m_net_training import training_utils
import tensorflow as tf
from keras.layers import Conv2D,Lambda,Input,BatchNormalization,Activation,AveragePooling2D,GlobalAveragePooling2D,Flatten,ReLU,Dense,multiply,Reshape,Concatenate,MaxPooling2D,Dropout,Multiply,LeakyReLU
from keras.activations import sigmoid
from keras.models import Model
from keras.utils import plot_model
from keras import regularizers


def white_norm(input): # this is used for normalizing whitish of image, kind of works as increase saturation, contrast & reduce brightness,.... Only included in first layer
  return (input - tf.constant(127.5)) / 128.0

def BRA(input):
  bn = BatchNormalization()(input)
  activtn = Activation('relu')(bn)
  return AveragePooling2D(pool_size=(2,2),strides=(2,2))(activtn)

def SE_BLOCK(input,using_SE=True,r_factor=2):
  channels_count = input.get_shape()[-1]
  act = GlobalAveragePooling2D()(input)
  fc1 = Dense(channels_count//r_factor,activation='relu')(act)
  scale = Dense(channels_count,activation='sigmoid')(fc1)
  return multiply([scale,input])


def build_shared_plain_network(input_height,input_width,input_channels,using_white_norm=True, using_SE=True):
  # design base model here
  input_shape = (input_height,input_width,input_channels)
  input_image = Input(shape=input_shape)
  if using_white_norm:
    wn = Lambda(white_norm,name='white_norm')(input_image)
    conv1 = Conv2D(32,(3,3),use_bias=False)(wn)
  else:
    conv1 = Conv2D(32,(3,3),use_bias=False)(input_image)

  block1 = BRA(input=conv1) # img/filters size reduced by half cuz Avgpooling2D with stride=(2,2)
  block1 = SE_BLOCK(input=block1, using_SE=using_SE)

  conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2")(block1)  # param 9248 = 32 * 32 * 3 * 3 + 32
  block2 = BRA(conv2) # img size half
  block2 = SE_BLOCK(block2, using_SE)  # put the se_net after BRA which achived better!!!!

  conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3")(block2)  # 9248
  block3 = BRA(conv3)# img size half
  block3 = SE_BLOCK(block3, using_SE)

  #at this point img/filters size is 4x4 so cannot reduce more, so using only B(BN) & R(relu) & excluded A (avgpool)
  conv4 = Conv2D(32,(3,3),use_bias=False,name='conv4')(block3)
  block4 = BatchNormalization()(conv4)
  block4 = Activation(activation='relu')(block4)
  block4 = SE_BLOCK(block4, using_SE)


  conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)  # 1024 + 32
  conv5 = SE_BLOCK(conv5, using_SE)  # r=16 Not as effective as conv5

  # understand this -1 in reshape
  flat_conv = Reshape((-1,))(conv5)
  # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
  # fc or pooling or any ohter operation
  #shape = map(int, conv5.get_shape()[1:])
  #shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

  baseModel = Model(inputs=input_image, outputs=[flat_conv])
  return baseModel



def build_net(Categories=12, input_height=64, input_width=64, input_channels=3, using_white_norm=True, using_SE=True):
    #building basic plain compact model for basic feature extrction
    base_model = build_shared_plain_network(input_height=input_height,input_width=input_width,input_channels=input_channels, using_white_norm=using_white_norm, using_SE=using_SE)

    x1 = Input(shape=(input_height, input_width, input_channels))
    x2 = Input(shape=(input_height, input_width, input_channels))
    # x3 = Input(shape=(input_height, input_width, input_channels))

    y1 = base_model(x1)
    y2 = base_model(x2)
    # y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2])
    bulk_feat = Dense(Categories, use_bias=True, activity_regularizer=regularizers.l1(0), activation='softmax', name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    #gender = Dense(2, activation=softmax, activity_regularizer=regularizers.l2(0), name="gender")(cfeat)

    # model = Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat]) 
    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in range(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(inputs=[x1, x2], outputs=[age, bulk_feat])

#--------------------- MY CUSTOM MODEL... SRAE -------------------

def CBRA(inputs):
  x = Conv2D(32,(3,3))(inputs)
  x = BatchNormalization(axis=-1)(x)
  x = Activation('relu')(x)
  x = SE_BLOCK(input=x)
  x = Dropout(0.2)(x)
  x = Conv2D(32,(5,5))(x)
  x = BatchNormalization(axis=-1)(x)
  x = Activation('relu')(x)
  x = SE_BLOCK(input=x)
  x_layer = AveragePooling2D(2,2)(x)
  return x_layer

# stream 2 module
def CBTM(inputs):
  s = Conv2D(2,(1,1))(inputs)
  s = BatchNormalization(axis=-1)(s)
  s = Activation('tanh')(s)
  s = SE_BLOCK(input=s)
  s = Conv2D(4,(2,2))(inputs)
  s = BatchNormalization(axis=-1)(s)
  s = Activation('tanh')(s)
  s = SE_BLOCK(input=s)
  s = Dropout(0.2)(s)
  s = Conv2D(8,(3,3))(inputs)
  s = BatchNormalization(axis=-1)(s)
  s = Activation('tanh')(s)
  s = SE_BLOCK(input=s)
  # s = Dropout(0.2)(s)    
  s = Conv2D(16,(5,5))(s)
  s = BatchNormalization(axis=-1)(s)
  s = Activation('tanh')(s)
  s = SE_BLOCK(input=s)
  s_layer = MaxPooling2D(2,2)(s)   
  return s_layer

def PB(inputs):
  #s_layer2_mix = Flatten()(inputs)
  s_layer2_mix = GlobalAveragePooling2D()(inputs)
  s_layer2_mix = Dense(6,activation='relu')(s_layer2_mix)
  s_layer2_mix = Dense(12,activation='relu',activity_regularizer=regularizers.l1(0))(s_layer2_mix)
  s_layer2_mix = Dropout(0.2)(s_layer2_mix)
  s_layer2_mix = Dense(3,activation='relu')(s_layer2_mix)
  return s_layer2_mix

def first_embd(x1,isPB_Block=False):
  x = CBRA(x1)
  y = CBTM(x1)
  if isPB_Block:
    x = PB(x)
    y = PB(y)
    first_embd = Concatenate(axis=-1)([x,y])
    return first_embd
  else:
    return x,y

def second_embd(x1,isPB_Block=False):
  x,y = first_embd(x1,False)
  x = CBRA(x)
  y = CBTM(y)
  if isPB_Block:
    x = PB(x)
    y = PB(y)
    scnd_embd = Concatenate(axis=-1)([x,y])
    return scnd_embd
  else:
    return x,y

def third_embd(x1):
  x,y = second_embd(x1,False)
  x = CBRA(x)
  y = CBTM(y)
  x = PB(x)
  y = PB(y)
  scnd_embd = Concatenate(axis=-1)([x,y])
  return scnd_embd

def build_ssr(Categories, input_height, input_width, input_channels, using_white_norm=True, using_SE=True):
  input_X = Input(shape=(input_height, input_width, input_channels))
  w1 = Lambda(white_norm,name='white_norm')(input_X)
  #--------- STREAM-1 ---------
  frst_embd = first_embd(w1,isPB_Block=True)
  scnd_embd = second_embd(w1,isPB_Block=True)
  thrd_embd = third_embd(w1)
  cfeat = Concatenate(axis=-1)([frst_embd, scnd_embd,thrd_embd])
  return Model(inputs=[input_X], outputs=[cfeat],name='SRAE')


def build_model(Categories=12, input_height=64, input_width=64, input_channels=3, using_white_norm=True, using_SE=True):
  x1 = Input(shape=(input_height, input_width, input_channels))
  x2 = Input(shape=(input_height, input_width, input_channels))
  x3 = Input(shape=(input_height, input_width, input_channels))
  ssr_model = build_ssr(Categories=Categories,input_height=input_height,input_width=input_width,input_channels=input_channels, using_white_norm=using_white_norm, using_SE=using_SE)

  y1 = ssr_model(x1)
  y2 = ssr_model(x2)
  y3 = ssr_model(x3)

  cfeat = Concatenate(axis=-1)([y1, y2,y3])
  bulk_feat = Dense(Categories, use_bias=True, activity_regularizer=regularizers.l1(0), activation='softmax', name="W1")(cfeat)
  # m = Dropout(0.2)(bulk_feat)
  age = Dense(1, name="age")(bulk_feat)  
  return Model(inputs=[x1,x2,x3], outputs=[age, bulk_feat])
