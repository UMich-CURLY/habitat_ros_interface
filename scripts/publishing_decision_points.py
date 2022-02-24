import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import csv

rospy.init_node("get_points",anonymous=False)
rate = rospy.Rate(10)
_pub_markers = rospy.Publisher("~points", MarkerArray, queue_size = 0)

final_plan = []

with open('data/maps/path_1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        final_plan.append(list(map(float,row)))


def main():
    msg = MarkerArray()
    counter = 0
    for wp in final_plan:
        marker = Marker()
        marker.id = counter;
        marker.header.frame_id = "decision_frame"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.pose.position.x = wp[0];
        marker.pose.position.y = wp[1];
        marker.pose.position.z = 0.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.05;
        marker.color.a = 1.0; 
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        counter = counter+1
        msg.markers.append(marker)
    rospy.loginfo("Publishing Markers...")
    _pub_markers.publish(msg)
    rospy.loginfo("Publishing Markers...")
    
    while not rospy.is_shutdown():
        rate.sleep()



if __name__ == "__main__":
	main()