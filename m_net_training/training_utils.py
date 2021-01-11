import numpy as np
import cv2
import pickle
import random
import keras.backend as K
import matplotlib.pyplot as plt

# adding random box in image with random colored pixels, it makes model generic????
def random_img_erasing(img, dropout=0.3, aspect=(0.5, 2), area=(0.06, 0.10)):
    # https://arxiv.org/pdf/1708.04896.pdf
    if 1 - random.random() > dropout:
        return img
    img = img.copy()
    height, width = img.shape[:-1]
    aspect_ratio = np.random.uniform(*aspect)
    area_ratio = np.random.uniform(*area)
    img_area = height * width * area_ratio
    dwidth, dheight = np.sqrt(img_area * aspect_ratio), np.sqrt(img_area * 1 / aspect_ratio) 
    xmin = random.randint(0, height)
    ymin = random.randint(0, width)
    xmax, ymax = min(height, int(xmin + dheight)), min(width, int(ymin + dwidth))
    img[xmin:xmax,ymin:ymax,:] = np.random.random_integers(0, 256, (xmax-xmin, ymax-ymin, 3))
    return img


def get_margin_right_left(landmarks,gap_margin):
  # gap_margin *=2 # because will be calculated on 2 sides
  # calculate percentange_of_right & percentange_of_left distance from total_distance
  total_distance,nose_to_left = landmarks[16][0]-landmarks[0][0], landmarks[30][0] - landmarks[0][0]
  percent_left = nose_to_left*100/total_distance
  # calculate margin values for right & left side.
  left_margin = round(gap_margin * percent_left/100)
  #confirmation of left+right == total_margin
  right_margin = gap_margin-left_margin
  return right_margin,left_margin

def get_margin_up_down_split(gap_margin,down_split=0.3):
  # calculate margin values for up & down side.
  down_margin = round(gap_margin * down_split)
  #confirmation of left+right == total_margin
  up_margin = gap_margin-down_margin
  return up_margin,down_margin


def evaluate_face_box(x1,y1,x2,y2,landmarks, force_align =False):
  left_lm = landmarks[17][0] # left side of face
  right_lm = landmarks[26][0] # right side of face
  top_lm = landmarks[24][1] # top side of face
  down_lm = landmarks[57][1] # bottom side of face
  
  
  if (left_lm < x1) or (right_lm > x2) or force_align:
    avg_offset_h = int (((x1 - left_lm) + (x2 - right_lm))/2)
    x1 -= avg_offset_h
    x2 -= avg_offset_h

  if (top_lm < y1) or (down_lm > y2) or force_align:
    avg_offset_v = int (((y1 - top_lm) + (y2 - down_lm))/2)
    y1 -= avg_offset_v
    y2 -= avg_offset_v

  return x1,y1,x2,y2


def gen_boundbox(box, landmark):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    xmin, ymin, xmax, ymax = box # box is [ymin, xmin, ymax, xmax]
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark.parts()[30].x, landmark.parts()[30].y) # calculating nose center point, so the triple boxes will be cropped according to nose point
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # Contains the smallest frame
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])


def gen_triple_face_box(box,landmarks,percent_margin=30):
  xmin, ymin, xmax, ymax = box 
  h = xmax - xmin
  #calculate gap value for bigger box
  gap_margin = round(h * percent_margin/100)
  
  # inner-box
  left_margin,right_margin = get_margin_right_left(landmarks,gap_margin)
  # print(left_margin,right_margin)
  if right_margin != 0:
    if 0.66 < left_margin/right_margin < 1.55:
      xmin, ymin, xmax, ymax = evaluate_face_box(xmin, ymin, xmax, ymax, landmarks,force_align=True) 
    else:
      xmin, ymin, xmax, ymax = xmin-5, ymin-5, xmax+5, ymax+5
  box_array = [[(xmin,ymin),(xmax,ymax)]]
  
  # middle box
  left_margin,right_margin = get_margin_right_left(landmarks,gap_margin) # calculate gap_margin right and left
  up_margin , down_margin  = get_margin_up_down_split(gap_margin)
  new_X =  int(xmin - left_margin)
  new_Y = int(ymin - up_margin)
  new_X2 = int(xmax + right_margin)
  new_Y2 = int(ymax + down_margin)
  new_X,new_Y,new_X2,new_Y2 = evaluate_face_box(new_X,new_Y,new_X2,new_Y2,landmarks)
  box_array.append([(new_X,new_Y),(new_X2,new_Y2)])
  # outer box
  gap_margin = gap_margin*2 # because 3rd box will be further outside
  left_margin,right_margin = get_margin_right_left(landmarks,gap_margin) # calculate gap_margin right and left
  up_margin , down_margin  = get_margin_up_down_split(gap_margin)
  new_X = int(xmin - left_margin)
  new_Y = int(ymin - up_margin)
  new_X2 =int(xmax + right_margin)
  new_Y2 =int(ymax + down_margin)
  new_X,new_Y,new_X2,new_Y2 = evaluate_face_box(new_X,new_Y,new_X2,new_Y2,landmarks)
  box_array.append([(new_X,new_Y),(new_X2,new_Y2)])
  return np.array(box_array) 

def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, category)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < category:
            embed[idx+1] = right_prob
        return embed
    return np.array(age_split(age_label))


def image_transform(row,dropout,target_img_shape,random_erasing=False,random_enforcing=False):
  # read image from buffer then decode
  img = np.frombuffer(row["image"], np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  #add random noise	
  if random_erasing:	  # normalize to the range 0-1
    img = random_img_erasing(img,dropout=dropout)

  # normalize to the range 0-1
  # img = img.astype('float32')
  # img = img/255.0
  # centering images (centering then normlizing make it center around 0.5)
  # centering images (centering before normlizing make it center around 0 (distributed around pos, neg): better for training but not for display images)
  # global_mean = img.mean()
  # img -= global_mean
  # local_mean => calculated mean for separated channels and centering each around it's mean
    # loval_mean = pixels.mean(axis=(0,1), dtype='float64')
    # img -= global_mean

  #STANDARDIZATION (preferred)
  # means = img.mean(axis=(0,1), dtype='float64') # mean of separate channels
  # stds = img.std(axis=(0,1), dtype='float64') # std of separate channels
  # img = (img - means) / stds

  # get trible box (out,middle,inner) and crop image from these boxes then
  face_lm = pickle.loads(row['landmarks'],encoding="bytes")
  face_box = pickle.loads(row['org_box'],encoding="bytes")
  triple_box = gen_boundbox(face_box,face_lm)
  # triple_box= gen_triple_face_box(face_box,face_lm,percent_margin=-10)

  #if contains a negative value, add padding to image.
  # if triple_box.min() < 0:
  #   padding = np.abs(triple_box.min()) + 1
  # else:
  padding = 20
  
  img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
  tripple_cropped_imgs = []
  # rr  = random.randint(0, 200)
  img_new = img.copy()
  # for box in pickle.loads(row['trible_box'],encoding="bytes"): # deserializing object which we converted to binary format using myNumArray.dump() method
  for box in triple_box: # deserializing object which we converted to binary format using myNumArray.dump() method
    h_min, w_min = box[0] #xmin, ymin
    h_max, w_max = box[1] #xmax, ymax
    # print('img shape {} & trible box {} '.format(img.shape,box))
    # crop image according to box size and add to list
    triple_box_cropped = cv2.resize(img_new[w_min+padding:w_max+padding, h_min+padding: h_max+padding], target_img_shape) # cropping image
    # triple_box_cropped = triple_box_cropped # resize according to size we want
    tripple_cropped_imgs.append(triple_box_cropped)
  
  # # list of [image_path,age]
  # f,axarr = plt.subplots(nrows=1,ncols=3,figsize=(10,10))
  # for i in range(3):
  #     m = tripple_cropped_imgs[i].copy()
  #     axarr[i].imshow(m)
  # # f.savefig("/content/my_gen/{}-{}.jpg".format(row['age'],rr))

  # tripple_cropped_imgs = []
  # img_new = img.copy()
  # for box in pickle.loads(row['trible_box'],encoding="bytes"): # deserializing object which we converted to binary format using myNumArray.dump() method
  # # for box in triple_box: # deserializing object which we converted to binary format using myNumArray.dump() method
  #   h_min, w_min = box[0] # xmin,ymin
  #   h_max, w_max = box[1] #xmax, ymax
  #   # print('img shape {} & trible box {} '.format(img.shape,box))
  #   # crop image according to box size and add to list
  #   triple_box_cropped = cv2.resize(img_new[w_min+padding:w_max+padding, h_min+padding: h_max+padding], target_img_shape) # cropping image
  #   # triple_box_cropped = cv2.resize(triple_box_cropped, target_img_shape) # resize according to size we want
  #   tripple_cropped_imgs.append(triple_box_cropped)  

  # # list of [image_path,age]
  # f,axarr = plt.subplots(nrows=1,ncols=3,figsize=(10,10))
  # for i in range(3):
  #     m = tripple_cropped_imgs[i].copy()
  #     axarr[i].imshow(m)
  # f.savefig("/content/his_gen/{}-{}.jpg".format(row['age'],rr))

  cascad_imgs = np.array(tripple_cropped_imgs)
  if random_enforcing:	
      flag = random.randint(0, 3)	
      contrast = random.uniform(0.5, 2.5)	
      bright = random.uniform(-50, 50)	
      rotation = random.randint(-15, 215)	
      cascad_imgs = [image_enforcing(x, flag, contrast, bright, rotation) for x in cascad_imgs]

  return cascad_imgs    


def img_and_age_data_generator(dataset_df,batch_size = 32, category=12, interval=10,imgs_shape =(64,64), random_erasing=False,random_enforcing=False, dropout = 0.2):
  dataset_df = dataset_df.reset_index(drop=True)
  df_count = len(dataset_df)
  while True:
    idx = np.random.permutation(df_count) # it will return a list of numbrs (0-df_count), in randomnly arranged
    start = 0
    while start+batch_size < df_count:
      idx_to_get = idx[start:start+batch_size] # making a list of random indexes, to get them from dataset
      current_batch = dataset_df.iloc[idx_to_get] # fetching a sub_df, which is our batch
      #load imgs, transform& create a list
      img_List = []
      two_point_ages = [] # list for 2_point_rep of ages
      for index,row in current_batch.iterrows(): #iterate over batch to load & transform each img
        # load and transform image
        img = image_transform(row, dropout=dropout,target_img_shape=imgs_shape,random_erasing=random_erasing,random_enforcing=random_enforcing)
        img_List.append(img)
        # make 2_point_represenation(list) of age
        two_point_rep = two_point(int(row.age), category, interval)
        two_point_ages.append(two_point_rep)    

      img_nparray = np.array(img_List) # converting image list to np
      two_point_ages_nparray = np.array(two_point_ages) # converting to np
      out = [current_batch.age.to_numpy(),two_point_ages_nparray] # making list of age_array & 2point_reprseation_array

      # print(len(two_point_ages_nparray[0]))

      yield [img_nparray[:,2], img_nparray[:,1],img_nparray[:,0]], out # return batch
      start += batch_size # update start point, for next batch

def image_enforcing(img, flag, contrast, bright, rotation):
    if flag & 1:  # trans hue
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=bright)
    elif flag & 2:  # rotation
        height, width = img.shape[:-1]
        matRotate = cv2.getRotationMatrix2D((height // 2, width // 2), rotation, 1) # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img, matRotate, (height, width))
    elif flag & 4:  # flp 翻转
        img = cv2.flip(img, 1)
    return img

############### Model Related things
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    # copy from https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-6, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ (total_num / ff if ff != 0 else 0.0) for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        print(classes_w_t2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)
        return fianal_loss
    return focal_loss_fixed

# ----------------------CYCLIC LEARNING RATE----------------------------------------------
#https://arxiv.org/abs/1506.01186

from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())



#------------------- LR_FINDER ----------------------------


# import the necessary packages
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
class LearningRateFinder:
	def __init__(self, model, stopFactor=4, beta=0.98):
		# store the model, stop factor, and beta value (for computing
		# a smoothed, average loss)
		self.model = model
		self.stopFactor = stopFactor
		self.beta = beta
		# initialize our list of learning rates and losses,
		# respectively
		self.lrs = []
		self.losses = []
		# initialize our learning rate multiplier, average loss, best
		# loss found thus far, current batch number, and weights file
		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None

	def reset(self):
		# re-initialize all variables from our constructor
		self.lrs = []
		self.losses = []
		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None


	def is_data_iter(self, data):
		# define the set of class types we will check for
		iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
			 "DataFrameIterator", "Iterator", "Sequence","ImgAgeGenrator"]
		# return whether our data is an iterator
		return data.__class__.__name__ in iterClasses


	def on_batch_end(self, batch, logs):
		# grab the current learning rate and add log it to the list of
		# learning rates that we've tried
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)
		# grab the loss at the end of this batch, increment the total
		# number of batches processed, compute the average average
		# loss, smooth it, and update the losses list with the
		# smoothed value
		l = logs["loss"]
		self.batchNum += 1
		self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
		smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
		self.losses.append(smooth)
		# compute the maximum loss stopping factor value
		stopLoss = self.stopFactor * self.bestLoss
		# check to see whether the loss has grown too large
		if self.batchNum > 1 and smooth > stopLoss:
			# stop returning and return from the method
			self.model.stop_training = True
			return
		# check to see if the best loss should be updated
		if self.batchNum == 1 or smooth < self.bestLoss:
			self.bestLoss = smooth
		# increase the learning rate
		lr *= self.lrMult
		K.set_value(self.model.optimizer.lr, lr)



	def find(self, trainData, startLR, endLR, epochs=None,
		stepsPerEpoch=None, batchSize=32, sampleSize=2048,
		verbose=1):
		# reset our class-specific variables
		self.reset()
		# determine if we are using a data generator or not
		useGen = True# self.is_data_iter(trainData)
		# if we're using a generator and the steps per epoch is not
		# supplied, raise an error
		if useGen and stepsPerEpoch is None:
			msg = "Using generator without supplying stepsPerEpoch"
			raise Exception(msg)
		# if we're not using a generator then our entire dataset must
		# already be in memory
		elif not useGen:
			# grab the number of samples in the training data and
			# then derive the number of steps per epoch
			numSamples = len(trainData[0])
			stepsPerEpoch = np.ceil(numSamples / float(batchSize))
		# if no number of training epochs are supplied, compute the
		# training epochs based on a default sample size
		if epochs is None:
			epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

		# compute the total number of batch updates that will take
		# place while we are attempting to find a good starting
		# learning rate
		numBatchUpdates = epochs * stepsPerEpoch
		# derive the learning rate multiplier based on the ending
		# learning rate, starting learning rate, and total number of
		# batch updates
		self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
		# create a temporary file path for the model weights and
		# then save the weights (so we can reset the weights when we
		# are done)
		self.weightsFile = tempfile.mkstemp()[1]
		self.model.save_weights(self.weightsFile)
		# grab the *original* learning rate (so we can reset it
		# later), and then set the *starting* learning rate
		origLR = K.get_value(self.model.optimizer.learning_rate)
		K.set_value(self.model.optimizer.learning_rate, startLR)

		# construct a callback that will be called at the end of each
		# batch, enabling us to increase our learning rate as training
		# progresses
		callback = LambdaCallback(on_batch_end=lambda batch, logs:
			self.on_batch_end(batch, logs))
		# check to see if we are using a data iterator
		if useGen:
			self.model.fit(
				x=trainData,
				steps_per_epoch=stepsPerEpoch,
				epochs=epochs,
				verbose=verbose,
				callbacks=[callback])
		# otherwise, our entire training data is already in memory
		else:
			# train our model using Keras' fit method
			self.model.fit(
				x=trainData[0], y=trainData[1],
				batch_size=batchSize,
				epochs=epochs,
				callbacks=[callback],
				verbose=verbose)
		# restore the original model weights and learning rate
		self.model.load_weights(self.weightsFile)
		K.set_value(self.model.optimizer.lr, origLR)

	def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
		# grab the learning rate and losses values to plot
		lrs = self.lrs[skipBegin:-skipEnd]
		losses = self.losses[skipBegin:-skipEnd]
		# plot the learning rate vs. loss
		plt.plot(lrs, losses)
		plt.xscale("log")
		plt.xlabel("Learning Rate (Log Scale)")
		plt.ylabel("Loss")
    # if the title is not empty, add it to the plot
		if title != "":
			plt.title(title)
		plt.savefig('/content/fig.jpg')








