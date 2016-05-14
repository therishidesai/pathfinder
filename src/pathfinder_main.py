#!/usr/bin/env python
import roslib
roslib.load_manifest('pathfinder')
import sys
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
from matplotlib import pyplot as plt
from pathfinder.msg import CameraData

class pathfinder:

    cv_image = None
    cv_depth_image = None
    thresh = None

    def __init__(self):
        self.bridge = CvBridge()
        cam_data = rospy.Subscriber("camera_data", CameraData, self.rgb_filter)

    def rgb_filter(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data.rgb_image)
        cv_image = cv2.flip(cv_image, 0)
        depth_thresh = self.bridge.imgmsg_to_cv2(data.thresh_image)
        #depth_thresh = cv2.resize(depth_thresh, (640,480), interpolation=cv2.INTER_AREA)
        cv_image_mask = cv2.bitwise_and(cv_image, cv_image, mask=depth_thresh)
        grayscale_thresh = self.gray_filter(cv_image_mask)
        ret, contours, hierarchy = cv2.findContours(grayscale_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = self.contour_filter(contours)
        img = cv2.drawContours(cv_image, self.cnt, -1, (0,255,0), 3)
        cv2.fillPoly(cv_image, pts =[self.cnt], color=(0,255,0))
        cv2.imshow("Image Window", cv_image)
        #cv2.imshow("Grayscale", grayscale_thresh)
        cv2.waitKey(3)

    def hsv_filter(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(cv_image.shape[:2], np.uint8)
        mask[200:500, 200:400] = 255
        masked_img = cv2.bitwise_and(cv_image,cv_image,mask = mask)
        #hist_mask = cv2.calcHist([cv_image],[0],mask,[256],[0,256])
        mean_hsv = cv2.mean(hsv, mask = mask)
        lower_hsv = [self.calc_hsv(mean_hsv[0],1), self.calc_hsv(mean_hsv[1],1), self.calc_hsv(mean_hsv[2],1)]
        upper_hsv = [self.calc_hsv(mean_hsv[0],0), self.calc_hsv(mean_hsv[1],0), self.calc_hsv(mean_hsv[2],0)]
        np_lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        np_upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        thresh = cv2.inRange(hsv, np_lower_hsv, np_upper_hsv)
        return thresh

    def gray_filter(self, cv_image):
        grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(cv_image.shape[:2], np.uint8)
        mask[300:500, 150:450] = 255
        masked_img = cv2.bitwise_and(cv_image,cv_image,mask = mask)
        mean_gray = cv2.mean(grayscale, mask=mask)
        #cv2.imshow("Mask", mask)
        thresh = cv2.inRange(grayscale, mean_gray[0]-20, mean_gray[0]+20)
        return thresh

    def contour_filter(self, contours):
        return max(contours, key=cv2.contourArea)

    def calc_hsv(self, hsv, over_under):
        if over_under==1:
            return int(hsv-(float(hsv/2.0)))
        else:
            return int(hsv+(float(hsv/2.0)))

def main():
    rospy.init_node('pathfinder', anonymous=True)
    pf = pathfinder()
    #print "test"

    #cv2.namedWindow('Image Window')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
