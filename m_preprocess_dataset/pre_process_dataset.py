import pandas as pd
import numpy as np
import cv2
# import json # to serialize objects, so can be stored as string in pandas's feather file
# import matplotlib.pyplot as plt
from pathlib import Path
import dlib
from pose import get_rotation_angle

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


def gen_equal_boundbox(box,gap_margin=30):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    xmin, ymin, xmax, ymax = box # box is [ymin, xmin, ymax, xmax]
    w, h = xmax - xmin, ymax - ymin

    percentMargin = gap_margin/100 # 30% margin
    margin_y = int(h * percentMargin)
    margin_x = int(h * percentMargin)

    # calculating new coordinates
    new_X =  xmin - margin_x 
    new_Y = ymin - margin_y
    new_X2 = xmax + int(margin_x) # mutliply by 2 because x is going backwards by same value, so adding 2 times margin
    new_Y2 = ymax + int(margin_y) # mutliply by 2 because y is going Upwards by same value

    return np.array([[(xmin,ymin),(xmax,ymax)], # original box
                    [(new_X,new_Y),(new_X2,new_Y2)]]) #  outer box


def detect_faces_and_landmarks(image):
  face_rect_list = detector(image)
  img_face_count = len(face_rect_list) # number of faces found in image
  if img_face_count < 1:
    return 0,[],[] # no face found, so return 

  xmin, ymin, xmax, ymax = face_rect_list[0].left() , face_rect_list[0].top(), face_rect_list[0].right(), face_rect_list[0].bottom() # face_rect is dlib.rectangle object, so extracting values from it
  
  # make a landmarks_list of all faces detected in image
  lmarks_list = dlib.full_object_detections()
  for face_rect in face_rect_list:
    lmarks_list.append(predictor(image, face_rect)) # getting landmarks as a list of objects
  
  return img_face_count,np.array([xmin, ymin, xmax, ymax]), lmarks_list

def loadData_preprocessData_and_makeDataFrame():
  properties_list = []

  for index,series in dataset_meta.iterrows():
    image_path = str(dataset_base_path.joinpath(series.imgPath)) # get image path
    # print(image_path,type(image_path))
    # break
    try:
      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      face_count,_,lmarks_list = detect_faces_and_landmarks(image) # Detect face & landmarks
      if face_count != 1:
        raise Exception("more than 1 or no face found in image ",image_path )
      #########################CropFace + genBox###################################
      #extract_image_chips will crop faces from image according to size & padding and align them in upright position and return list of them
      cropped_faces = dlib.get_face_chips(image, lmarks_list, padding=0.6)  # aligned face with padding 0.4 in papper
      image = cropped_faces[0] # must be only 1 face, so getting it.
      _,face_rect_box, lmarks_list = detect_faces_and_landmarks(image) # Detect face from cropped image
      first_lmarks = lmarks_list[0] # getting first face's rectangle box and landmarks 
      double_box = gen_boundbox(face_rect_box,first_lmarks) # get 2 face boxes for nput into network, as reauired in paper
      ####################################Save image to check #######################################
      test_img = cropped_faces[0].copy()
      if index % 5000 == 0:
        cv2.imwrite('/content/saved{}_original.jpg'.format(index),test_img)
        # for bbox in double_box:
        #   bbox = bbox
        #   h_min, w_min = bbox[0]
        #   h_max, w_max = bbox[1]
        #   cv2.rectangle(test_img, (h_min,w_min), (h_max,w_max),(255,0,0),2)
        #   cv2.imwrite('/content/saved{}_original.jpg'.format(index),test_img)
      ###########################################################################
      # if (double_box < 0).any():
      #   raise Exception('Some part of face is out of image ')
      # face_pitch, face_yaw, face_roll = get_rotation_angle(image, first_lmarks) # gen face rotation for filtering
    except Exception as ee:        
      # print('index ',index,': exption ',ee)
      properties_list.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]) # add null dummy values to current row & skill this iteration
      continue
      
    # everything processed succefuly, now serialize values and save them
    status, buf = cv2.imencode(".jpg", image)
    image_buffer = buf.tostring()
    #dumping with `pickle` much faster than `json` (np.dumps is pickling)
    face_rect_box_serialized = face_rect_box.dumps()  # [xmin, ymin, xmax, ymax] : Returns the pickle(encoding to binary format (better than json)) of the array as a string. pickle.loads or numpy.loads will convert the string back to an array
    landmarks_list = np.array([[point.x,point.y] for point in first_lmarks.parts()]) # Same converting landmarks (face_detection_object) to array so can be converted to json
    face_landmarks_serialized = landmarks_list.dumps()#json.dumps(landmarks_list,indent = 2)  # y1..y5, x1..x5
    
    # adding everything to list
    properties_list.append([image_path,series.age,series.gender,image_buffer,face_rect_box_serialized,face_landmarks_serialized])
    if index%500 == 0:
      print(index,'images added processed')
  processed_dataset_df = pd.DataFrame(properties_list,columns=['image_path','age','gender','image','org_box','landmarks'])
  # some filtering on df
  processed_dataset_df = processed_dataset_df.dropna()
  processed_dataset_df = processed_dataset_df[(processed_dataset_df.age >= 0) & (processed_dataset_df.age <= 100)]
  
  return processed_dataset_df # returning now (just in case need to return), maybe later remove...


# save processed dataset_df to feather format
def save(chunkSize=5000):
    print('df ssize save  ',len(Dataset_DF))
    dataframe = Dataset_DF.reset_index()
    chunk_start = 0
    while(chunk_start < len(Dataset_DF)):
        dir_path = dataset_base_path.joinpath(dataset_name + "_" + str(int(chunk_start / chunkSize)) + ".feather")
        tmp_pd = dataframe[chunk_start:chunk_start + chunkSize].copy().reset_index()
        tmp_pd.to_feather(dir_path)
        chunk_start += chunkSize
        print('succesfully saved as feather to ',dir_path)



def rectify_data():
    sample = []
    max_nums = 500.0
    for x in range(100):
        age_set = Dataset_DF[Dataset_DF.age == x]
        cur_age_num = len(age_set)
        if cur_age_num > max_nums:
            age_set = age_set.sample(frac=max_nums / cur_age_num, random_state=2007)
        sample.append(age_set)
    Dataset_DF = pd.concat(sample, ignore_index=True)
    Dataset_DF.age = Dataset_DF.age 
    print(Dataset_DF.groupby(["age", "gender"]).agg(["count"]))

################################## GLOBAL PARAMS ##############################################
#initiate face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/RP/detect/shape_predictor_68_face_landmarks.dat")

# creating dummy DF, later will process all images to make it real df
Dataset_DF = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks"])

# define all parameters here
dataset_base_path = Path('/content/RP/dataset/')
dataset_meta = pd.read_csv('/content/RP/dataset/Dataset/dataset-meta.csv')
extra_padding = 0.55

if __name__ == "__main__":

    # init_dataset_meta_csv() # convert meta.mat to meta.csv
    Dataset_DF = loadData_preprocessData_and_makeDataFrame()
    save() # save preprocessed dataset as .feather in  dataset_directory_path
