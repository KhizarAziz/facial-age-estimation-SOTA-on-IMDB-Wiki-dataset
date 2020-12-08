import numpy as np
import cv2
import pickle
import random
import keras.backend as K

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
  return left_margin,right_margin

def get_margin_up_down_split(gap_margin,down_split=0.3):
  # calculate margin values for up & down side.
  down_margin = round(gap_margin * down_split)
  #confirmation of left+right == total_margin
  up_margin = gap_margin-down_margin
  return up_margin,down_margin


def gen_triple_face_box(box,landmarks,percent_margin=30):
  xmin, ymin, xmax, ymax = box 
  h = xmax - xmin
  #calculate gap value for bigger box
  gap_margin = round(h * percent_margin/100)
  # inner-box
  box_array = [[(xmin,ymin),(xmax,ymax)]]
  # middle box
  left_margin,right_margin = get_margin_right_left(landmarks,gap_margin) # calculate gap_margin right and left
  up_margin , down_margin  = get_margin_up_down_split(gap_margin)
  new_X =  int(xmin - left_margin)
  new_Y = int(ymin - up_margin)
  new_X2 = int(xmax + right_margin)
  new_Y2 = int(ymax + down_margin)
  box_array.append([(new_X,new_Y),(new_X2,new_Y2)])
  # outer box
  gap_margin = gap_margin*2 # because 3rd box will be further outside
  left_margin,right_margin = get_margin_right_left(landmarks,gap_margin) # calculate gap_margin right and left
  up_margin , down_margin  = get_margin_up_down_split(gap_margin)
  new_X = int(xmin - left_margin)
  new_Y = int(ymin - up_margin)
  new_X2 =int(xmax + right_margin)
  new_Y2 =int(ymax + down_margin)
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
  if random_erasing:
    img = random_img_erasing(img,dropout=dropout)
  # get trible box (out,middle,inner) and crop image from these boxes then
  face_lm = pickle.loads(row['landmarks'],encoding="bytes")
  face_box = pickle.loads(row['org_box'],encoding="bytes")
  triple_box= gen_triple_face_box(face_box,face_lm,percent_margin=45)

  #if contains a negative value, add padding to image.
  if triple_box.min() < 0:
    padding = np.abs(triple_box.min()) + 1
  else:
    padding = 0
  
  img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
  tripple_cropped_imgs = []
  # for box in pickle.loads(row['trible_box'],encoding="bytes"): # deserializing object which we converted to binary format using myNumArray.dump() method
  for box in triple_box: # deserializing object which we converted to binary format using myNumArray.dump() method
    h_min, w_min = box[0] # xmin,ymin
    h_max, w_max = box[1] #xmax, ymax
    # print('img shape {} & trible box {} '.format(img.shape,box))
    # crop image according to box size and add to list
    triple_box_cropped = img[w_min+padding:w_max+padding, h_min+padding: h_max+padding] # cropping image
    triple_box_cropped = cv2.resize(triple_box_cropped, target_img_shape) # resize according to size we want
    tripple_cropped_imgs.append(triple_box_cropped)
    #image augmentaion (hue, contrast,rotation etc) if needed
  cascad_imgs = np.array(tripple_cropped_imgs)
  if random_erasing:
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

      yield [img_nparray[:,0], img_nparray[:,1],img_nparray[:,2]], out # return batch
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







