#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:35:33 2018

@author: lochappy
"""

import numpy as np
from filterpy.kalman import KalmanFilter
import cv2

kf = KalmanFilter(dim_x=7, dim_z=4)
kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],  
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

kf.R[2:,2:] *= 10.
kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
kf.P *= 10.
kf.Q[-1,-1] *= 0.01
kf.Q[4:,4:] *= 0.01


cv_kf = cv2.KalmanFilter(7,4)
#F
cv_kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],  
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]],np.float32)

#H
cv_kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]],np.float32)

# R
cv_kf.measurementNoiseCov = np.array([[  1.,   0.,   0.,   0.],
                                      [  0.,   1.,   0.,   0.],
                                      [  0.,   0.,  10.,   0.],
                                      [  0.,   0.,   0.,  10.]],np.float32)

# P
cv_kf.errorCovPost = np.array([[10,0,0,0,0,0,0],
                               [0,10,0,0,0,0,0],
                               [0,0,10,0,0,0,0],
                               [0,0,0,10,0,0,0],  
                               [0,0,0,0,10000,0,0],
                               [0,0,0,0,0,10000,0],
                               [0,0,0,0,0,0,10000]],np.float32)

# Q
cv_kf.processNoiseCov = np.array([[1,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0],
                                  [0,0,1,0,0,0,0],
                                  [0,0,0,0.01,0,0,0],  
                                  [0,0,0,0,0.01,0,0],
                                  [0,0,0,0,0,0.01,0],
                                  [0,0,0,0,0,0,0.0001]],np.float32)

#x
cv_kf.statePost = np.array([[ 0.],
                            [ 0.],
                            [ 0.],
                            [ 0.],
                            [ 0.],
                            [ 0.],
                            [ 0.]],np.float32)

