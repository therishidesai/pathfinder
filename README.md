# pathfinder
https://www.youtube.com/watch?v=mPOzu3il9TM Finshed project video.

The program first takes the depth image in the rosbag and removes areas that are too far away to travel on. Then using this filter as a mask I put the RGB image on that mask and then convert it to grayscale. Then I take the area 3 inches infront of the robot and calculate the average grayscale value. With the average grayscale value I threshold the grayscale image. Then I contour the thresholded image and take the largest contour and fill it. Finally the path is displayed in green on the final image.
