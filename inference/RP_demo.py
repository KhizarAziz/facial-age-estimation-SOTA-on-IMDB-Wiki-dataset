import os
import cv2
import numpy as np
import argparse
# from SSRNET_model import SSR_net, SSR_net_general
import sys
import timeit
from moviepy.editor import *
import keras
from keras import backend as K
import tensorflow as tf
import pdb

def gen_equal_boundbox(box,percent_margin=20):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    xmin, ymin, w, h = box # box is [ymin, xmin, ymax, xmax]
    xmin, ymin, xmax, ymax = xmin, ymin, xmin+w, ymin+h
    # xmin, ymin, xmax, ymax = xmin-10, ymin-10, xmax+10, ymax+10
    # w, h = xmax - xmin, ymax - ymin

    box_array = [[(xmin,ymin),(xmax,ymax)]] # inner-box

    # middle box
    margin = int(h * percent_margin/100) # 15% margin
    new_X =  xmin - margin 
    new_Y = ymin - margin
    new_X2 = xmax + margin 
    new_Y2 = ymax + margin 
    box_array.append([(new_X,new_Y),(new_X2,new_Y2)])

    # outer box
    margin = int(margin*-0.9) # 30% margin
    new_X =  xmin - margin 
    new_Y = ymin - margin
    new_X2 = xmax + margin 
    new_Y2 = ymax + margin 
    box_array.append([(new_X,new_Y),(new_X2,new_Y2)])

    return np.array(box_array) 

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot):
    
    #for i, d in enumerate(detected):
    for i, (x,y,w,h) in enumerate(detected):
        #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)
        
        faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        
        faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)
        
    
    start_time = timeit.default_timer()



    if len(detected) > 0:
        # pdb.set_trace()
        trible_box = gen_equal_boundbox(detected[0],percent_margin=3)
        tri_imgs = []
        # temp_imgs = []
        for bbox in trible_box:
            bbox = bbox #+ padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            cropped_img = cv2.resize(input_img[w_min:w_max, h_min:h_max], (64, 64))
            # temp_imgs.append(cropped_img)
            tri_imgs.append(np.expand_dims(cropped_img,axis=0))        
        # predict ages and genders of the detected faces
        predicted_ages = model.predict(tri_imgs)
        # predicted_genders = model_gender.predict(faces)
        
    # print("age is : ",predicted_ages)
    # draw results
    for i, (x,y,w,h) in enumerate(detected):
        #label = "{}~{}, {}".format(int(predicted_ages[i]*4.54),int((predicted_ages[i]+1)*4.54),
        #                       "F" if predicted_genders[i][0] > 0.5 else "M")
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        # gender_str = 'male'
        # if predicted_genders[i]<0.5:
        #     gender_str = 'female'
        try:
            label = "{}".format(int(predicted_ages[i]))
        
            draw_label(input_img, (x1, y1), label)
        except Exception as e:
            print('Exxcptn: ',e)
    
    elapsed_time = timeit.default_timer()-start_time
    time_network = time_network + elapsed_time
    
    
    
    start_time = timeit.default_timer()

    #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("result", input_img)
        
    
    elapsed_time = timeit.default_timer()-start_time
    time_plot = time_plot + elapsed_time

    return input_img,time_network,time_plot

def main():
    K.set_learning_phase(0) # make sure its testing mode
    # weight_file = "../pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    # weight_file_gender = "../pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    
    face_cascade = cv2.CascadeClassifier('face_cascade.xml')

    try:
        os.mkdir('./img')
    except OSError:
        pass

    # load model and weights
    img_size = 64
    # stage_num = [3,3,3]
    # lambda_local = 1
    # lambda_d = 1
    # model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    # model.load_weights(weight_file)

    # model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    # model_gender.load_weights(weight_file_gender)

    model = keras.models.load_model('model.h5', custom_objects={'tf': tf})
    print('model loadded')
    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
    
    

    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 10 # every 5 frame do 1 detection and network forward propagation
    ad = 0.5

    while True:
        # get video frame
        ret, input_img = cap.read()

        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        
        if img_idx==1 or img_idx%skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0
            
            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
            start_time = timeit.default_timer()
            detected = face_cascade.detectMultiScale(gray_img, 1.1)
            elapsed_time = timeit.default_timer()-start_time
            time_detection = time_detection + elapsed_time
            faces = np.empty((len(detected), img_size, img_size, 3))

            model_gender = None
            input_img,time_network,time_plot = draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot)
            cv2.imwrite('img/'+str(img_idx)+'.png',input_img)
            
        else:
            input_img,time_network,time_plot = draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot)
        
        #Show the time cost (fps)
        print('avefps_time_detection:',1/time_detection)
        print('avefps_time_network:',skip_frame/time_network)
        print('avefps_time_plot:',skip_frame/time_plot)
        print('===============================')
        key = cv2.waitKey(1)
        


if __name__ == '__main__':
    main()