import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import habitat
import habitat_sim.bindings as hsim
import magnum as mn
import csv
from habitat.utils.visualizations import maps
# %matplotlib inline
from matplotlib import pyplot as plt

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# function to display the topdown map
from PIL import Image
rospy.init_node("get_points",anonymous=False)

def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.5):
	points_topdown = []
	bounds = pathfinder.get_bounds()
	for point in points:
		# convert 3D x,z to topdown x,y
		px = (point[0] - bounds[0][0]) / meters_per_pixel
		py = (point[2] - bounds[0][2]) / meters_per_pixel
		points_topdown.append(np.array([px, py]))
	return points_topdown



class tour_planner:
	selection_done = False
	_sensor_rate = 50  # hz
	_r = rospy.Rate(_sensor_rate)
	def __init__(self):
		env_config_file="configs/tasks/pointnav_rgbd.yaml"
		self.env = habitat.Env(config=habitat.get_config(env_config_file))
		self._pub_plan = rospy.Publisher("~global_plan", Path, queue_size=1)
		self._pub_markers = rospy.Publisher("~points", MarkerArray, queue_size = 1)
		self.selected_points = []
		self.selected_points_3d = []
		self.final_plan = []
		meters_per_pixel =0.05
		self.topdown_map = maps.get_topdown_map(
				self.env._sim.pathfinder, 0.0, meters_per_pixel=meters_per_pixel
			)
		recolor_map = np.array(
			[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
		)
		self.topdown_map = recolor_map[self.topdown_map]
		# self.generate_plan()

	def generate_distance_matrix(self):
		distance_matrix = []
		for start_point in self.selected_points_3d:
			distances = []
			for end_point in self.selected_points_3d:
				path = hsim.ShortestPath()
				path.requested_start = start_point
				path.requested_end = end_point
				found_path = False
				print(start_point, end_point)
				counter = 0
				while(found_path==False and counter<30):
					counter = counter+1
					found_path = self.env._sim.pathfinder.find_path(path)
				geodesic_distance = path.geodesic_distance
				distances.append(geodesic_distance)
			distance_matrix.append(distances)
		return distance_matrix

	def create_data_model(self):
		"""Stores the data for the problem."""
		data = {}
		distance_matrix = self.generate_distance_matrix()	    
		data['distance_matrix'] = distance_matrix  # yapf: disable
		data['num_vehicles'] = 1
		data['depot'] = 0
		return data	

	def generate_plan(self):
		data = self.create_data_model()
		manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
											   data['num_vehicles'], data['depot'])
		routing = pywrapcp.RoutingModel(manager)
		def distance_callback(from_index, to_index):
			"""Returns the distance between the two nodes."""
			# Convert from routing variable Index to distance matrix NodeIndex.
			from_node = manager.IndexToNode(from_index)
			to_node = manager.IndexToNode(to_index)
			return data['distance_matrix'][from_node][to_node]

		transit_callback_index = routing.RegisterTransitCallback(distance_callback)
		routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
		search_parameters = pywrapcp.DefaultRoutingSearchParameters()
		search_parameters.first_solution_strategy = (
			routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
		solution = routing.SolveWithParameters(search_parameters)
		print('Objective: {} miles'.format(solution.ObjectiveValue()))
		index = routing.Start(0)
		plan_output = 'Route for vehicle 0:\n'
		plan_order = []
		route_distance = 0
		while not routing.IsEnd(index):
			plan_output += ' {} ->'.format(manager.IndexToNode(index))
			plan_order.append(int(format(manager.IndexToNode(index))))
			self.final_plan.append(self.selected_points[int(format(manager.IndexToNode(index)))])
			previous_index = index
			index = solution.Value(routing.NextVar(index))
			route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
		plan_output += ' {}\n'.format(manager.IndexToNode(index))
		print(plan_order)
		self.final_plan.append(self.selected_points[0])
		print(self.final_plan)
		plan_output += 'Route distance: {}miles\n'.format(route_distance)
		self.publish_plan()
		self.publish_markers()

	def publish_markers(self):
		msg = MarkerArray()
		counter = 0
		for wp in self.final_plan:
			marker = Marker()
			marker.id = counter;
			marker.header.frame_id = "world"
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.SPHERE
			marker.pose.position.x = wp[0];
			marker.pose.position.y = wp[1];
			marker.pose.position.z = 0.0;
			marker.pose.orientation.x = 0.0;
			marker.pose.orientation.y = 0.0;
			marker.pose.orientation.z = 0.0;
			marker.pose.orientation.w = 1.0;
			marker.scale.x = 0.1;
			marker.scale.y = 0.1;
			marker.scale.z = 0.01;
			marker.color.a = 1.0; 
			marker.color.r = 1.0;
			marker.color.g = 0.0;
			marker.color.b = 0.0;
			counter = counter+1
			msg.markers.append(marker)
		rospy.loginfo("Publishing Markers...")
		self._pub_markers.publish(msg)


	def publish_plan(self):
		msg = Path()
		msg.header.frame_id = "world"
		msg.header.stamp = rospy.Time.now()
		for wp in self.final_plan:
			pose = PoseStamped()
			pose.pose.position.x = wp[0]
			pose.pose.position.y = wp[1]
			pose.pose.position.z = 0.0
			pose.pose.orientation.x = 0.0
			pose.pose.orientation.y = 0.0
			pose.pose.orientation.z = 0.0
			pose.pose.orientation.w = 1.0
			msg.poses.append(pose)
		rospy.loginfo("Publishing Plan...")
		self._pub_plan.publish(msg)
		# self._r.sleep()

	def callback(self, point, tour_plan):
		xy_points = {}
		navigable = []
		with open('./scripts/handpicked_points_3d.csv', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			for row in spamreader:
				map_points_3d = list(map(float,row))
				if(self.env._sim.pathfinder.is_navigable(map_points_3d)):
					self.selected_points.append(list(map(float,row)))
					self.selected_points_3d.append(list(map(float,row)))
					navigable.append(self.env._sim.pathfinder.is_navigable(map_points_3d))
		self.selected_points = np.array(self.selected_points)
		self.selected_points = convert_points_to_topdown(self.env._sim.pathfinder, self.selected_points)
		xy_points["points"] = self.selected_points
		xy_points["navigable"] = navigable
		tour_plan.generate_plan()
		# if (tour_plan.selection_done == True):
		# 	print("Already selected points, going to generate plan now!")
		# 	tour_plan.generate_plan()
		# 	return
		# tour_plan.selected_points.append([point.point.x, point.point.y, 0.0])
		# tour_plan.selection_done = bool(input("Want to finish selection?"))
		


def main():
	tour_plan = tour_planner()
	rospy.Subscriber("/clicked_point", PointStamped,tour_plan.callback, tour_plan,queue_size=1)
	rospy.spin()
	# # define a list capturing how long it took
	# # to update agent orientation for past 3 instances
	# # TODO modify dt_list to depend on r1
	# dt_list = [0.009, 0.009, 0.009]
	# while not rospy.is_shutdown():
	# 	tour_plan.generate_plan()
	#     start_time = time.time()
	#     # cv2.imshow("bc_sensor", my_env.observations['bc_sensor'])
	#     # cv2.waitKey(100)
	#     # time.sleep(0.1)
	#     my_env.update_orientation()

	#     dt_list.insert(0, time.time() - start_time)
	#     dt_list.pop()
	#     my_env.set_dt(sum(dt_list) / len(dt_list))


if __name__ == "__main__":
	main()