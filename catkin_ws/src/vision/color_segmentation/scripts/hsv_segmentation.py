#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2024-2
# COLOR SEGMENTATION USING HSV
#
# Instructions:
# Write the code necessary to detect and localize the 'pringles'
# or 'drink' using only a hsv color segmentation.
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#


import numpy
import cv2
import ros_numpy
import rospy
import math
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
from vision_msgs.srv import RecognizeObject, RecognizeObjectResponse

NAME = "Frías Hernández Camille Emille Román"

def segment_by_color(img_bgr, points, obj_name):
    global mask, kernel
    img_x, img_y, x,y,z = 0,0,0,0,0
    #x=columns, y = rows
    # TODO:
    # - Assign lower and upper color limits according to the requested object:
    #   If obj_name == 'pringles': [25, 50, 50] - [35, 255, 255]
    #   otherwise                : [10,200, 50] - [20, 255, 255]
    # - Change color space from BGR to HSV with cvtColor().
    #   Check online documentation for cv2.cvtColor(imageBGR, cv2.COLOR_BGR2HSV) function
    # - Determine the pixels whose color is in the selected color range.
    #   Check online documentation for cv2.inRange(hsv, upper, lower) upper y lower son los arreglos para el mínimo y máximo de los colores 
    # - Calculate the centroid of all pixels in the given color range (ball position).
    #   Check online documentation for cv2.findNonZero and cv2.mean
    # - Calculate the centroid of the segmented region in the cartesian space
    #   using the point cloud 'points'. Use numpy array notation to process the point cloud data.
    #   Example: 'points[240,320][1]' gets the 'y' value of the point corresponding to
    #   the pixel in the center of the image.
    #
    # Establece los límites de color en HSV según el objeto
    if obj_name == 'pringles':
        lower = numpy.array([25, 50, 50])
        upper = numpy.array([35, 255, 255])
    else:
        lower = numpy.array([10, 200, 50])
        upper = numpy.array([20, 255, 255])

    # Convierte la imagen de BGR a HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Segmenta la imagen según el rango de color
    mask = cv2.inRange(hsv, lower, upper)

    # Realiza una erosión seguida de una dilatación para limpiar la máscara
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Encuentra los píxeles en la región segmentada
    pixels = cv2.findNonZero(mask)
    if pixels is not None:
        # Calcula el centroide en la imagen
        moments = cv2.moments(mask)
        img_x = int(moments['m10'] / moments['m00'])
        img_y = int(moments['m01'] / moments['m00'])

        # Obtén las coordenadas del centroide en el espacio cartesiano usando la nube de puntos
        x = points[img_y, img_x][0]
        y = points[img_y, img_x][1]
        z = points[img_y, img_x][2]

    return [img_x, img_y, x, y, z]

def callback_find_object(req):
    global pub_point, img_bgr
    print("Trying to find object: " + req.name)
    arr = ros_numpy.point_cloud2.pointcloud2_to_array(req.point_cloud)
    rgb_arr = arr['rgb'].copy()
    rgb_arr.dtype = numpy.uint32
    r = numpy.asarray(((rgb_arr >> 16) & 255), dtype='uint8')
    g = numpy.asarray(((rgb_arr >>  8) & 255), dtype='uint8')
    b = numpy.asarray(((rgb_arr      ) & 255), dtype='uint8')
    img_bgr = cv2.merge((b,g,r))
    [r, c, x, y, z] = segment_by_color(img_bgr, arr, req.name)
    resp = RecognizeObjectResponse()
    resp.recog_object.header.frame_id = 'realsense_link'
    resp.recog_object.header.stamp    = rospy.Time.now()
    resp.recog_object.pose.position.x = x
    resp.recog_object.pose.position.y = y
    resp.recog_object.pose.position.z = z
    pub_point.publish(PointStamped(header=resp.recog_object.header, point=Point(x=x, y=y, z=z)))
    cv2.circle(img_bgr, (int(r), int(c)), 20, [0, 255, 0], thickness=3)
    return resp

def main():
    global pub_point, img_bgr, mask, kernel
    print("COLOR SEGMENTATION - " + NAME)
    rospy.init_node("color_segmentation")
    erosion_shape=rospy.get_param("~kernel",0)
    erosion_size=rospy.get_param("~size",1)
    kernel = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1))
    
    rospy.Service("/vision/obj_reco/detect_and_recognize_object", RecognizeObject, callback_find_object)
    pub_point = rospy.Publisher('/detected_object', PointStamped, queue_size=10)
    img_bgr = numpy.zeros((480, 640, 3), numpy.uint8)
    mask = numpy.zeros((480, 640, 3), numpy.uint8)
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        cv2.imshow("Color Segmentation", img_bgr)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)
        loop.sleep()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

