#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
from matplotlib import pyplot as plt
from pathfinder.msg import CameraData

class camera:

    rgb_image = None
    cv_depth_thresh = None

    def __init__(self):
        self.bridge = CvBridge()
        pub = rospy.Publisher("camera_data", CameraData)
        depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        cv_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        data = self.send_camera_data(self.rgb_image, self.cv_depth_thresh)
        pub.publish(data)

    def get_cv_image(self):
        return self.cv_image

    def get_cv_depth_image(self):
        return self.cv_depth_image

    def rgb_callback(self, data):
        self.rgb_image = data

    def depth_callback(self, data):
        cv_depth_image=0
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print e

        cv_depth_image = cv2.flip(cv_depth_image, 0)

        self.cv_depth_thresh = depth_filter(cv_depth_image)

    def depth_filter(self, depth_image):
        cv_depth_image = depth_image
        cv_depth_image = cv2.resize(cv_depth_image, (640,480), interpolation = cv2.INTER_AREA)
        lower =0
        upper =4000
        thresh = cv2.inRange(cv_depth_image, lower, upper)
        thresh = cv2.resize(thresh, (640,480), interpolation = cv2.INTER_AREA)
        #ret, thresh = cv2.threshold(cv_depth_image, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def send_camera_data(self, rgb_image, thresh_image):
        cd = CameraData()
        cd.rgb_image = rgb_image
        cd.thresh_image = self.bridge.cv2_to_imgmsg(thresh_image, "passthrough")
        return cd

def main():
    rospy.init_node('camera_node', anonymous=True)
    c = camera()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
