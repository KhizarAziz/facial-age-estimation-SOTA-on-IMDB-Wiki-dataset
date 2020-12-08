import pandas as pd
import numpy as np
import cv2
import json # to serialize objects, so can be stored as string in pandas's feather file
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from datetime import datetime
import dlib
from IPython.core.debugger import set_trace
# from pose import get_rotation_angle,get_landmarks,warp_im,transformation_from_points,get_dummy_refrence_face

# process it to detect faces, detect landmarks, align, & make 3 sub boxes which will be used in next step to feed into network
# save dataset as pandas,feather & imencode for size efficiency

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

def gen_equal_boundbox(box,gap_margin=20):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    xmin, ymin, xmax, ymax = box # box is [ymin, xmin, ymax, xmax]
    w, h = xmax - xmin, ymax - ymin

    box_array = [[(xmin,ymin),(xmax,ymax)]] # inner-box

    # middle box
    margin = int(h * gap_margin/100) # 15% margin
    new_X =  xmin - margin 
    new_Y = ymin - margin
    new_X2 = xmax + margin 
    new_Y2 = ymax + margin 
    box_array.append([(new_X,new_Y),(new_X2,new_Y2)])

    # outer box
    margin = int(margin*2) # 30% margin
    new_X =  xmin - margin 
    new_Y = ymin - margin
    new_X2 = xmax + margin 
    new_Y2 = ymax + margin 
    box_array.append([(new_X,new_Y),(new_X2,new_Y2)])

    return np.array(box_array) 

def get_margin_right_left(landmarks,gap_margin):
  # gap_margin *=2 # because will be calculated on 2 sides
  # calculate percentange_of_right & percentange_of_left distance from total_distance
  total_distance,nose_to_left = landmarks[16].x-landmarks[0].x, landmarks[30].x - landmarks[0].x
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


def calculate_age(dob, image_capture_date):
  birth = datetime.fromordinal(max(int(dob) - 366, 1))
  # assume the photo was taken in the middle of the year
  if birth.month < 7:
      return image_capture_date - birth.year
  else:
      return image_capture_date - birth.year - 1

class Process_WIKI_IMDB():  
  def __init__(self,base_path,dataset_name,extra_padding):
    #init meta.mat file
    self.dataset_name = dataset_name
    meta_file_name = dataset_name+'.mat'
    self.base_path = Path(base_path)
    self.meta_file_path = self.base_path.joinpath(meta_file_name)
    self.extra_padding = extra_padding
    self.Dataset_Df = pd.DataFrame() # creating dummy DF, later will process all images to make it real df


  
  def meta_to_csv(self,dataset_name,test_num=-1):
    # make list of all columns in meta.mat file
    meta = loadmat(self.meta_file_path)
    full_path = [p[0] for p in meta[dataset_name][0, 0]["full_path"][0][:test_num]]
    dob = meta[dataset_name][0, 0]["dob"][0][:test_num]  # Matlab serial date number
    mat_gender = meta[dataset_name][0, 0]["gender"][0][:test_num]
    photo_taken = meta[dataset_name][0, 0]["photo_taken"][0][:test_num]  # year
    face_score = meta[dataset_name][0, 0]["face_score"][0][:test_num]
    second_face_score = meta[dataset_name][0, 0]["second_face_score"][0][:test_num]
    age = [calculate_age(dob[i],photo_taken[i] ) for i in range(len(dob))] # calculate age using dob and taken date
    # make a dataframe dataset
    Dataset_DF = pd.DataFrame({"full_path": full_path, "age": age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
    # Completing image paths from root to image
    Dataset_DF['full_path'] = str(self.base_path)+'/'+Dataset_DF['full_path']
    # save dataframe as csv
    Dataset_DF.to_csv(self.base_path.joinpath(self.dataset_name+'.csv'),index=False)


  def detect_faces_and_landmarks(self,image):
    face_rect_list = detector(image,1)
    img_face_count = len(face_rect_list) # number of faces found in image
    if img_face_count < 1:
      # print('no faces detectedd,,   ',len(face_rect_list))
      return 0,[],[] # no face found, so return 

    xmin, ymin, xmax, ymax = face_rect_list[0].rect.left() , face_rect_list[0].rect.top(), face_rect_list[0].rect.right(), face_rect_list[0].rect.bottom() # face_rect is dlib.rectangle object, so extracting values from it
    
    # make a landmarks_list of all faces detected in image
    lmarks_list = dlib.full_object_detections()
    for face_rect in face_rect_list:
      lmarks_list.append(predictor(image, face_rect.rect)) # getting landmarks as a list of objects
    
    return img_face_count,np.array([xmin, ymin, xmax, ymax]), lmarks_list

  def loadData_preprocessData_and_makeDataFrame(self):
    meta_dataframe = pd.read_csv(self.base_path.joinpath(self.dataset_name+'.csv'))
    # init lists of all properties gonna be saved
    properties_list = []
    # no_face_list = []
    # loop through meta.csv for all images
    for index,series in meta_dataframe.iterrows():
      # clear multiple faces
      image_path = series.full_path
      try:
        #filter out where second face != null (image have 2 faces)
        # if not pd.isnull(series.second_face_score):
        #   raise Exception("has second face -> image has 2 face ",image_path )
        if (series.age < 1) or (series.age > 100):
          raise Exception("Age out of bound")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image.shape[0] < 20:
          raise Exception("image Size is tooooo small") # images are (1,1,3).. dont know why
        # image = cv2.copyMakeBorder(image, self.extra_padding, self.extra_padding, self.extra_padding, self.extra_padding, cv2.BORDER_CONSTANT)
        face_count,_,lmarks_list = self.detect_faces_and_landmarks(image) # Detect face & landmarks
        if face_count > 1:
          raise Exception("face_count > 1")
        elif face_count < 1:
          # no_face_list.append([series.full_path,series.age])
          raise Exception("no face detected")
        # found exactly 1 face, so now process it
        #########################CropFace + genBox###################################
        #extract_image_chips will crop faces from image according to size & padding and align them in upright position and return list of them
        cropped_faces = dlib.get_face_chips(image, lmarks_list, padding=0.99)  # aligned face with padding 0.4 in papper
        # crop2 = dlib.get_face_chips(image, lmarks_list, padding=0.8,size=(64,64))  # aligned face with padding 0.4 in papper
        image = cropped_faces[0] # must be only 1 face, so getting it.
        _,face_rect_box, lmarks_list = self.detect_faces_and_landmarks(image) # Detect face from cropped image
        first_lmarks = lmarks_list[0] # getting first face's rectangle box and landmarks 
        # triple_box = gen_triple_face_box(face_rect_box,first_lmarks.parts()) # get 2 face boxes for nput into network, as reauired in paper
        ####################################Save image to check #######################################
        # test_img = cropped_faces[0]
        # if index % 5000 == 0:
        #   for bbox in triple_box:
        #     bbox = bbox
        #     h_min, w_min = bbox[0]
        #     h_max, w_max = bbox[1]
        #     cv2.rectangle(test_img, (h_min,w_min), (h_max,w_max),(255,0,0),2)
        #     cv2.imwrite('/content/saved{}_original.jpg'.format(index),image)
        #   print('test image saved /content/saved{}_original.jpg'.format(index))
        ###########################################################################
        # detect face landmarks again from cropped & align face.  (as positions of lmarks are changed in cropped image)

        # if (triple_box < 0).any():
        #   raise Exception('Some part of face is out of image ')
        
        # face_pitch, face_yaw, face_roll = get_rotation_angle(image, first_lmarks) # gen face rotation for filtering
      except Exception as ee:        
        # print('index ',index,': exption ',ee,series.full_path)
        # raise Exception(ee)
        properties_list.append([image_path,series.age,np.nan,np.nan,np.nan,np.nan]) # add null dummy values to current row & skill this iteration
        continue
        
      # everything processed succefuly, now serialize values and save them
      status, buf = cv2.imencode(".jpg", image)
      image_buffer = buf.tostring()
      #dumping with `pickle` much faster than `json` (np.dumps is pickling)
      face_rect_box_serialized = face_rect_box.dumps()  # [xmin, ymin, xmax, ymax] : Returns the pickle(encoding to binary format (better than json)) of the array as a string. pickle.loads or numpy.loads will convert the string back to an array
      # trible_boxes_serialized = triple_box.dumps() # 3 boxes of face as required in paper
      landmarks_list = np.array([[point.x,point.y] for point in first_lmarks.parts()]) # Same converting landmarks (face_detection_object) to array so can be converted to json
      face_landmarks_serialized = landmarks_list.dumps()#json.dumps(landmarks_list,indent = 2)  # y1..y5, x1..x5
      
      # adding everything to list
      properties_list.append([image_path,series.age,series.gender,image_buffer,face_rect_box_serialized,face_landmarks_serialized])
      if index%500 == 0:
        print(index,'images preprocessed')
    processed_dataset_df = pd.DataFrame(properties_list,columns=['image_path','age','gender','image','org_box','landmarks'])
    # processed_dataset_df.to_csv('/content/Full_Dataset.csv',index=False)
    # df = pd.DataFrame(no_face_list,columns=['image_path','age'])
    # df.to_csv('/content/no_face_found.csv')
    # some filtering on df
    processed_dataset_df = processed_dataset_df.dropna()
    processed_dataset_df = processed_dataset_df[(processed_dataset_df.age >= 0) & (processed_dataset_df.age <= 100)]
    processed_dataset_df.to_csv('/content/Filtered_Dataset.csv',index=False)
    self.Dataset_Df = processed_dataset_df
    return processed_dataset_df # returning now (just in case need to return), maybe later remove...



  # save processed dataset_df to feather format
  def save(self, chunkSize=5000):
      dataframe = self.Dataset_Df.reset_index()
      chunk_start = 0
      while(chunk_start < len(self.Dataset_Df)):
          dir_path = self.base_path.joinpath(self.dataset_name + "_" + str(int(chunk_start / chunkSize)) + ".feather")
          tmp_pd = dataframe[chunk_start:chunk_start + chunkSize].copy().reset_index()
          tmp_pd.to_feather(dir_path)
          chunk_start += chunkSize
          print('succesfully saved as feather to ',dir_path)


  def rectify_data(self):
      sample = []
      max_nums = 500.0
      for x in range(100):
          age_set = self.Dataset_Df[self.Dataset_Df.age == x]
          cur_age_num = len(age_set)
          if cur_age_num > max_nums:
              age_set = age_set.sample(frac=max_nums / cur_age_num, random_state=2007)
          sample.append(age_set)
      self.Dataset_Df = pd.concat(sample, ignore_index=True)
      self.Dataset_Df.age = self.Dataset_Df.age 
      print(self.Dataset_Df.groupby(["age", "gender"]).agg(["count"]))

Dataset_DF = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"])
#initiate face detector and predictor
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1('/content/C3AE/detect/mmod_human_face_detector.dat')
predictor = dlib.shape_predictor("/content/C3AE/detect/shape_predictor_68_face_landmarks.dat")

# define all parameters here
dataset_directory_path = '/content/C3AE/dataset/wiki_crop/'
dataset_name = 'wiki' # different dataset name means different sequence for loading etc
# image transform params (if require)
extra_padding = 0.55


if __name__ == "__main__":

  if dataset_name == 'wiki' or dataset_name == 'imdb': # because structure is same
    dataset_class_ref_object = Process_WIKI_IMDB(dataset_directory_path,dataset_name,extra_padding)
    dataset_class_ref_object.meta_to_csv(dataset_name) # convert meta.mat to meta.csv
    dataset_class_ref_object.loadData_preprocessData_and_makeDataFrame()
    dataset_class_ref_object.save() # save preprocessed dataset as .feather in  dataset_directory_path