#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from matplotlib import pyplot as plt

class pathfinder:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        #self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
    def rgb_callback(self, data):
        cv_image=0
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print e
        cv_image = cv2.flip(cv_image,0)
        grayscale_thresh = self.gray_filter(cv_image)
        ret, contours, hierarchy = cv2.findContours(grayscale_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = self.contour_filter(contours)
        img = cv2.drawContours(cv_image, cnt, -1, (0,255,0), 3)
        cv2.fillPoly(cv_image, pts =[cnt], color=(0,255,0))
        #edges = cv2.Canny(thresh,50,150,apertureSize = 3)
        #plt.subplot(224), plt.plot(hist_mask)
        cv2.imshow("Image Window", cv_image)
        #cv2.imshow("Mask", mask)
        cv2.imshow("Threshold", grayscale_thresh)
        #cv2.imshow("HSV", hsv)
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
        mask[200:500, 200:400] = 255
        masked_img = cv2.bitwise_and(cv_image,cv_image,mask = mask)
        mean_gray = cv2.mean(grayscale, mask=mask)
        thresh = cv2.inRange(grayscale, mean_gray[0]-20, mean_gray[0]+20)
        return thresh

    def contour_filter(self, contours):
        return max(contours, key=cv2.contourArea)

    def calc_hsv(self, hsv, over_under):
        if over_under==1:
            return int(hsv-(float(hsv/2.0)))
        else:
            return int(hsv+(float(hsv/2.0)))

def nothing(x):
    pass

def main():
    pf = pathfinder()
    rospy.init_node('pathfinder', anonymous=True)
    cv2.namedWindow('Image Window')
    cv2.createTrackbar('T','Image Window',0,255,nothing)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
