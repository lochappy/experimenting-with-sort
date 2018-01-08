#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:26:14 2018

@author: lochappy
"""

import numpy as np
import os.path
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from skimage import io
import time
import argparse

from sort import Sort
from detector import GroundTruthDetections

import cv2

def main():
    args = parse_args()
    display = args.display
    use_dlibTracker  = args.use_dlibTracker

    total_time = 0.0
    total_frames = 0

    # for disp
    if display:
        colours = (np.random.rand(32, 3)*255).astype(np.int)  # used only for display
        
    cap = cv2.VideoCapture('TownCentreXVID.avi')


    if not os.path.exists('output'):
        os.makedirs('output')
    out_file = 'output/townCentreOut.top'

    #init detector
    detector = GroundTruthDetections()

    #init tracker
    tracker =  Sort(use_dlib= use_dlibTracker) #create instance of the SORT tracker

    windowName = ""
    if use_dlibTracker:
        windowName = "Dlib Correlation Tracker"
        print "Dlib Correlation tracker activated!"
    else:
        windowName = "Kalman Tracker"
        print "Kalman tracker activated!"
        
    cv2.namedWindow(windowName,0)

    with open(out_file, 'w') as f_out:

        frames = detector.get_total_frames()
        for frame in range(0, frames):  #frame numbers begin at 0!
            # get detections
            detections = detector.get_detected_items(frame)

            total_frames +=1
            ret,img = cap.read()
            if ret == False:
                break

            start_time = time.time()
            #update tracker
            trackers = tracker.update(detections,img)

            cycle_time = time.time() - start_time
            total_time += cycle_time

            print('frame: %d...took: %3fs'%(frame,cycle_time))

            for d in trackers:
                f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], frame, 1, 1, d[0], d[1], d[2], d[3]))
                if (display):
                    #print colours[int(d[4] % 32), :]
                    cv2.rectangle(img,(int(d[0]), int(d[1])),(int(d[2]), int(d[3])),colours[int(d[4] % 32), :],2)
                    #label
                    cv2.putText(img,'%d' % (d[4]),(int(d[0]), int(d[1])),cv2.FONT_HERSHEY_SIMPLEX,1,colours[int(d[4] % 32), :],2)
                    if detections != []:#detector is active in this frame
                        cv2.putText(img,'DETECTOR',(5, 45),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4)

            cv2.imshow(windowName,img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            #if total_frames == 2:
            #    break
    #cv2.destroyWindow("img")
        
    #cv2.destroyAllWindows()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Experimenting Trackers with SORT')
    parser.add_argument('--NoDisplay', dest='display', help='Disables online display of tracker output (slow)',action='store_false')
    parser.add_argument('--dlib', dest='use_dlibTracker', help='Use dlib correlation tracker instead of kalman tracker',action='store_true')
    parser.add_argument('--save', dest='saver', help='Saves frames with tracking output, not used if --NoDisplay',action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()