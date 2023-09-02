import sys
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray
import tf2_ros
import tf2_geometry_msgs
import numpy as np




class speed_test():

    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.sub_1 = rospy.Subscriber("/sim/agent_poses", PoseArray, self.callback_1)
        # self.sub_2 = rospy.Subscriber("/sim/robot_pose", PoseArray, self.callback_2)
        self.prev_human_poses = None 
        self.human_poses = None
        self.prev_human_t_stamp = None
        self.human_t_stamp = None
        self.prev_robot_pose = None 
        self.robot_pose = None


    def callback_1(self, msg):
        first_pass = False
        if (not self.prev_human_poses):
            first_pass = True
        self.human_poses = []
        for pose in msg.poses:
            self.human_poses.append(pose)
            self.human_t_stamp = msg.header.stamp
        speeds = []
        if (not first_pass):
            i = 0
            dt = (self.human_t_stamp - self.prev_human_t_stamp).to_sec()
            for pose in self.human_poses:
                vel_x = pose.position.x - self.prev_human_poses[0].position.x
                vel_y = pose.position.y - self.prev_human_poses[0].position.y
                speed = np.linalg.norm(np.array([vel_x,vel_y]))/dt
                speeds.append(speed)
            print(np.mean(np.array(speeds)))
        self.prev_human_poses = self.human_poses
        self.prev_human_t_stamp = self.human_t_stamp

    def transform_pose(self, input_pose, from_frame, to_frame):

        # **Assuming /tf2 topic is being broadcasted

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now() - rospy.Duration(1)
        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, timeout = rospy.Duration(1))
            return output_pose_stamped

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("No Transform found?")
            raise
            return None 




if __name__ == "__main__":
    rospy.init_node("speed_test",anonymous=False)
    speed_tester = speed_test()
    while not rospy.is_shutdown():
        rospy.spin()
