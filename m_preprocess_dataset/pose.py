import os
import cv2
import numpy as np
import math


def quaterniondToRulerAngle(quaterniond):
    q = quaterniond
    y_sqrt = q.y ** 2
    # pitch
    t0 = 2 * (q.w * q.x + q.y * q.z)
    t1 = 1.0 - 2.0 * (q.x ** 2 + y_sqrt) 
    pitch = math.atan(t0 / t1)#math.atan2(t0, t1)

    # yaw
    t2 = 2 * (q.w * q.y - q.z * q.x)
    t2 = max(min(t2, 1), -1)
    yaw = math.asin(t2)

    # roll
    t3 = 2 * (q.w * q.z + q.x * q.y)
    t4 = 1 - 2 * (y_sqrt + q.z * q.z)
    roll = math.atan(t3 / t4)  #math.atan2(t3, t4)
    return pitch, yaw, roll

def tran_euler(rotation_vect):
    theta = cv2.norm(rotation_vect, cv2.NORM_L2)
    class Quation(object):
        def __init__(self, w, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
    quat = Quation(
        math.cos(theta / 2),
        math.sin(theta / 2) * rotation_vect[0][0] / theta,
        math.sin(theta / 2) * rotation_vect[1][0] / theta,
        math.sin(theta / 2) * rotation_vect[2][0] / theta
        )
    return map(lambda x: x / math.pi * 180, quaterniondToRulerAngle(quat))

def trans_landmarks(img, landmark_groups):
    result = []
    for lm in landmark_groups:
        landmarks = np.array([(lm[x], lm[5 + x],) for x in range(5)], dtype="double")
        for p in landmarks:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        result.append(get_rotation_angle(img, landmarks))
    return result

def get_rotation_angle(img, landmarks, draw=False):

    # you can read more about this method model on https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

    size = img.shape
    parts = landmarks.parts()

    #making an array of points(nose,chin,left-eye etc) from lm, which will be used to determine face angle
    image_points = np.array([
                                (parts[30].x, parts[30].y),     # Nose tip : point 30 in landmarks list
                                (parts[8].x, parts[8].y),     # Chin : same
                                (parts[36].x, parts[36].y),     # Left eye left corner
                                (parts[45].x, parts[45].y),     # Right eye right corne
                                (parts[48].x, parts[48].y),     # Left Mouth corner
                                (parts[54].x, parts[54].y)      # Right mouth corner
                            ], dtype="double")

    #3D locations of the same points : You also need the 3D location of the 2D feature points. You might be thinking that you need a 3D model of the person in the photo to get the 3D locations. Ideally yes, but in practice, you don’t. A generic 3D model will suffice. Where do you get a 3D model of a head from ? Well, you really don’t need a full 3D model. You just need the 3D locations of a few points in some arbitrary reference frame. In this tutorial, we are going to use the following 3D points.
    model_points = np.array([
              (0.0, 0.0, 0.0),             # Nose tip
              (0.0, -330.0, -65.0),        # Chin
              (-225.0, 170.0, -135.0),     # Left eye left corner
              (225.0, 170.0, -135.0),      # Right eye right corne
              (-150.0, -150.0, -125.0),    # Left Mouth corner
              (150.0, -150.0, -125.0)      # Right mouth corner
          ])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2,)
    camera_matrix = np.array([
             [focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]
         ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, trans_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    f_pitch, f_yaw, f_roll = tran_euler(rotation_vector)

    n_pitch = prod_trans_point((0, 0, 500.0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_yaw = prod_trans_point((200.0, 0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_roll = prod_trans_point((0, 500.0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)

    if draw:
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        cv2.line(img, p1, n_roll, (255, 0, 0), 2)
        cv2.line(img, p1, n_yaw, (0, 255, 0), 2)
        cv2.line(img, p1, n_pitch, (0, 0, 255), 2)
        cv2.putText(img, ("r:" + str(f_roll))[:6], n_roll, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("y:" + str(f_yaw))[:6], n_yaw, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("p:" + str(f_pitch))[:6], n_pitch, 1,1, (136, 97, 45), 2) 
    return f_pitch, f_yaw, f_roll

def prod_trans_point(p3d, rotation_vector, trans_vector, camera_matrix, dist_coeffs):
    plane_point, _ = cv2.projectPoints(np.array([p3d]), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    return (int(plane_point[0][0][0]), int(plane_point[0][0][1]))

################ face alignment #############

# we have already processed an image and got it's landmarks, so we dont need to download ref_image each time.
def get_dummy_refrence_face():
  ref_face_landmarks = np.matrix([[ 76, 134],
        [ 76, 147],
        [ 78, 161],
        [ 81, 173],
        [ 84, 186],
        [ 90, 196],
        [100, 203],
        [113, 207],
        [126, 208],
        [140, 207],
        [153, 203],
        [162, 195],
        [169, 184],
        [172, 172],
        [174, 158],
        [175, 144],
        [175, 130],
        [ 82, 126],
        [ 87, 120],
        [ 96, 119],
        [105, 120],
        [115, 123],
        [130, 123],
        [140, 119],
        [150, 117],
        [160, 118],
        [166, 123],
        [123, 130],
        [123, 137],
        [123, 145],
        [123, 153],
        [114, 160],
        [119, 161],
        [124, 162],
        [129, 161],
        [134, 159],
        [ 93, 133],
        [ 98, 129],
        [105, 129],
        [111, 133],
        [105, 134],
        [ 98, 135],
        [136, 133],
        [142, 128],
        [149, 128],
        [155, 131],
        [150, 133],
        [143, 133],
        [107, 177],
        [113, 173],
        [119, 171],
        [124, 172],
        [130, 171],
        [137, 173],
        [144, 177],
        [138, 181],
        [131, 182],
        [125, 183],
        [119, 183],
        [113, 182],
        [110, 177],
        [119, 176],
        [125, 176],
        [130, 175],
        [141, 177],
        [130, 175],
        [125, 176],
        [119, 175]])
  # landmarks of face are normally distributed like this.. i.e NOSE landmark points are from 27 to 35
  JAW_POINTS = list(range(0, 17))
  RIGHT_BROW_POINTS = list(range(17, 22))
  LEFT_BROW_POINTS = list(range(22, 27))
  NOSE_POINTS = list(range(27, 35))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_EYE_POINTS = list(range(42, 48))
  MOUTH_POINTS = list(range(48, 61))
  FACE_POINTS = list(range(17, 68))

  # Points used to line up the images.
  ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                                  RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

  return ref_face_landmarks,ALIGN_POINTS, (250, 250, 3) # this was shape of our image from which we got landmarks

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # converting points array to np.float64 to subtract (we cannot subtract normal int/float from an array... so convert to numpy float64 )
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # calcuating means i.e c1 & c2 are numpy float64 type of values
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1 # now can subtract
    points2 -= c2

    # taking standard deviation
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2


    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def get_landmarks(im,detector,predictor):
    rects = detector(im, 1)

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])








