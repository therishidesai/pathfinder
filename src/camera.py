#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
from matplotlib import pyplot as plt
import os
from pathfinder.msg import CameraData

rgb_image = None
cv_depth_thresh = None

'''def __init__():
    .bridge = CvBridge()
    .pub = rospy.Publisher("camera_data", CameraData)
    depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, .depth_callback)
    cv_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, .rgb_callback)
    if .rgb_image is not None:
        cv_image = .bridge.imgmsg_to_cv2(self.rgb_image, "passthrough")
        cv2.imshow("img", cv_image)
        cv2.waitKey(3)
        .data = self.send_camera_data(self.rgb_image, self.cv_depth_thresh)
        .pub.publish(self.data)
'''
def get_cv_image():
    return cv_image

def get_cv_depth_image():
    return cv_depth_image

def rgb_callback(data):
    global cv_depth_thresh
    bridge = CvBridge()
    rgb_image = data
    pub = rospy.Publisher("camera_data", CameraData)
    cd = CameraData()
    cd.rgb_image = rgb_image
    cd.thresh_image = bridge.cv2_to_imgmsg(cv_depth_thresh, "passthrough")
    pub.publish(cd)

def depth_callback(data):
    #print "test1"
    global cv_depth_thresh
    bridge = CvBridge()
    cv_depth_image=0
    try:
        cv_depth_image = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print e

    cv_depth_image = cv2.flip(cv_depth_image, 0)
    cv_depth_thresh = depth_filter(cv_depth_image)
    #cv2.imshow("depth", cv_depth_thresh)
    #cv2.waitKey(3)

def depth_filter(depth_image):
    cv_depth_image = depth_image
    cv_depth_image = cv2.resize(cv_depth_image, (640,480), interpolation = cv2.INTER_AREA)
    lower =0
    upper =6000
    thresh = cv2.inRange(cv_depth_image, lower, upper)
    thresh = cv2.resize(thresh, (640,480), interpolation = cv2.INTER_AREA)
    #ret, thresh = cv2.threshold(cv_depth_image, 127, 255, cv2.THRESH_BINARY)
    return thresh

def send_camera_data(rgb_image, thresh_image):
    #global cv_depth_thresh
    bridge = CvBridge()
    cd = CameraData()
    cd.rgb_image = rgb_image
    cd.thresh_image = bridge.cv2_to_imgmsg(thresh_image, "passthrough")
    return cd

def main():
    bridge = CvBridge()
    rospy.init_node('camera_node', anonymous=True)
    depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
    cv_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
