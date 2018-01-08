#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:57:25 2018

@author: lochappy
"""

import numpy as np
import cv2


'''Motion Model'''
class OpencvKalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,img=None):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = cv2.KalmanFilter(7,4)
    self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],
                                         [0,1,0,0,0,1,0],
                                         [0,0,1,0,0,0,1],
                                         [0,0,0,1,0,0,0],  
                                         [0,0,0,0,1,0,0],
                                         [0,0,0,0,0,1,0],
                                         [0,0,0,0,0,0,1]],np.float32)
    self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],
                                          [0,1,0,0,0,0,0],
                                          [0,0,1,0,0,0,0],
                                          [0,0,0,1,0,0,0]],np.float32)

    self.kf.measurementNoiseCov = np.array([[  1.,   0.,   0.,   0.],
                                            [  0.,   1.,   0.,   0.],
                                            [  0.,   0.,  10.,   0.],
                                            [  0.,   0.,   0.,  10.]],np.float32)
    #give high uncertainty to the unobservable initial velocities
    self.kf.errorCovPost = np.array([[10,0,0,0,0,0,0],
                                     [0,10,0,0,0,0,0],
                                     [0,0,10,0,0,0,0],
                                     [0,0,0,10,0,0,0],  
                                     [0,0,0,0,10000,0,0],
                                     [0,0,0,0,0,10000,0],
                                     [0,0,0,0,0,0,10000]],np.float32)
    self.kf.processNoiseCov = np.array([[1,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0],
                                        [0,0,1,0,0,0,0],
                                        [0,0,0,0.01,0,0,0],  
                                        [0,0,0,0,0.01,0,0],
                                        [0,0,0,0,0,0.01,0],
                                        [0,0,0,0,0,0,0.0001]],np.float32)

    tmp_bbox = convert_bbox_to_z(bbox)
    self.kf.statePost = np.array([[ tmp_bbox[0]],
                                  [ tmp_bbox[1]],
                                  [ tmp_bbox[2]],
                                  [ tmp_bbox[3]],
                                  [ 0.],
                                  [ 0.],
                                  [ 0.]],np.float32)
    self.time_since_update = 0
    self.id = OpencvKalmanBoxTracker.count
    OpencvKalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox,img=None):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    #print "update"
    if bbox != []:
        self.kf.correct(convert_bbox_to_z(bbox))

  def predict(self,img=None):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    
    #print "pred_state={}".format(self.kf.statePost.reshape(-1))
    if((self.kf.statePost[6][0]+self.kf.statePost[2][0])<=0):
        print "self.kf.statePost[6][0]={}".format(self.kf.statePost[6][0])
        self.kf.statePost[6][0] = 0.0
        print "self.kf.statePost[6][0]={}".format(self.kf.statePost[6][0])
      
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.statePost))
    return self.history[-1][0]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    #print "get_state {} ".format(convert_x_to_bbox(self.kf.statePost)[0])
    return convert_x_to_bbox(self.kf.statePost)[0]


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r],np.float32).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.],dtype=np.float32).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score],dtype=np.float32).reshape((1,5))