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
import sys
sys.path.insert(1, '/home/matching_routing/')
from MatchRouteWrapper import MatchRouteWrapper
from ResultVisualizer import ResultVisualizer
import helper


def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.5):
	points_topdown = []
	bounds = pathfinder.get_bounds()
	for point in points:
		# convert 3D x,z to topdown x,y
		px = (point[0] - bounds[0][0]) / meters_per_pixel
		py = (point[2] - bounds[0][2]) / meters_per_pixel
		points_topdown.append(np.array([px, py]))
	return points_topdown

def get_rgb_from_demand(demand):
	rgb = []
	if demand == 0:
		rgb = [0.0,0.0,0.0]
	elif demand <= 5:
		rgb = [1-(0.25*(demand-1)), 0.25*(demand-1), 0.0]
	else:
		rgb = [0.25*(demand-6), 0.0, 1-(0.25*(demand-6))]
	return rgb	


class tour_planner():
	selection_done = False
	
	def __init__(self, env):
		if (not rospy.core.is_initialized()):
			rospy.init_node("get_points",anonymous=False)
		_sensor_rate = 50  # hz
		self._r = rospy.Rate(_sensor_rate)
		self.env = env
		
		self._pub_markers = rospy.Publisher("~points", MarkerArray, queue_size = 1)
		self._pub_plan_initial = rospy.Publisher("~initial_global_plan", Path, queue_size=1)
		self._pub_markers_initial = rospy.Publisher("~selected_points", MarkerArray, queue_size = 1)
		self._pub_plan_3d_robot_1 = rospy.Publisher("robot_1/plan_3d", numpy_msg(Floats),queue_size = 1)
		self._pub_plan_robot_1 = rospy.Publisher("robot_1/global_plan", Path, queue_size=1)
		self._pub_plan_3d_robot_2 = rospy.Publisher("robot_2/plan_3d", numpy_msg(Floats),queue_size = 1)
		self._pub_plan_robot_2 = rospy.Publisher("robot_2/global_plan", Path, queue_size=1)
		self._pub_plan_3d_robot_3 = rospy.Publisher("robot_3/plan_3d", numpy_msg(Floats),queue_size = 1)
		self._pub_plan_robot_3 = rospy.Publisher("robot_3/global_plan", Path, queue_size=1)
		self.selected_points = []
		self.selected_points_3d = []
		self.final_plan_1 = []
		self.final_plan_3d_1 = []	## Use this to save and then finally display the shortest between nodes instead, TRIBHI!!!!!! 
		self.final_plan_2 = []
		self.final_plan_3d_2 = []	## Use this to save and then finally display the shortest between nodes instead, TRIBHI!!!!!! 
		self.navigable = []
		meters_per_pixel =0.05
		self.topdown_map = maps.get_topdown_map(
				self.env._sim.pathfinder, 0.0, meters_per_pixel=meters_per_pixel
			)
		recolor_map = np.array(
			[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
		)
		self.topdown_map = recolor_map[self.topdown_map]
		with open('./scripts/handpicked_points_3d.csv', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			for row in spamreader:
				map_points_3d = list(map(float,row))
				if(self.env._sim.pathfinder.is_navigable(map_points_3d)):
					self.selected_points.append(list(map(float,row)))
					self.selected_points_3d.append(list(map(float,row)))
					self.navigable.append(self.env._sim.pathfinder.is_navigable(map_points_3d))
		self.selected_points = np.array(self.selected_points)
		self.selected_points = convert_points_to_topdown(self.env._sim.pathfinder, self.selected_points)
		self.selected_points = np.array(self.selected_points)
		self.selected_points_3d = np.array(self.selected_points_3d)
		# self.demand_list = [0, 1, 1, 3, 6, 3, 6, 8, 8, 1, 2 , 1, 2, 6, 6, 8, 8, 9, 1, 3, 5, 2, 8, 4]
		self.demand_list = [8,6,3,1,6,2,4,1,3,7,9,5,9,5,2,6,6,9,7,5,5,3,2,0,0]
		self.node_num = len(self.selected_points)
		self.capacity = 80
		rospy.Subscriber("/clicked_point", PointStamped,self.callback,queue_size=1)
		self._r.sleep()

	def generate_distance_matrix(self):
		distance_matrix = []
		for start_point in self.selected_points_3d:
			distances = []
			# Check Size of the matrix
			for end_point in self.selected_points_3d:
				path = hsim.ShortestPath()
				path.requested_start = start_point
				path.requested_end = end_point
				found_path = False
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
		data['demands'] = self.demand_list
		data['vehicle_capacities'] = [self.capacity]
		data['depot'] = 0
		return data	

	def generate_plan(self):
		data = self.create_data_model()
		flag_verbose = True

		flag_initialize = 0 # 0: VRP, 1: random
		flag_solver = 1 # 0 GUROBI exact solver, 1: OrTool heuristic solver
		solver_time_limit = 300.0
		flag_uncertainty = False
		beta = 0.98
		sigma = 1.0

		scale_time = 1.0

		veh_num = 3
		node_num = len(self.selected_points)
		demand_penalty = 1000.0
		time_penalty = 10.0
		time_limit = 100 * scale_time
		human_num = 10
		human_choice = 5
		max_iter = 10
		max_human_in_team = np.ones(veh_num, dtype=int) * 4 # (human_num // veh_num + 5)

		node_seq = None
		# node_seq = [[0,1,2], [3,4]]

		angular_offset = 2.9

		# Initialize spacial maps
		node_pose = self.selected_points
		edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
		veh_speed = np.ones(veh_num, dtype=np.float64)*1.25/angular_offset
		edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1) * scale_time
		node_time = np.ones((veh_num,node_num), dtype=np.float64) * 5.5 * scale_time
		node_time[:, -2:] = 0.0
		print('node_time = ', node_time)
		if flag_uncertainty:
			edge_time_std = edge_time * sigma
			node_time_std = np.ones((veh_num,node_num), dtype=np.float64) * 1.5
		else:
			edge_time_std = None
			node_time_std = None

		# Initialize human selections
		global_planner = MatchRouteWrapper(veh_num, node_num, human_choice, human_num, max_human_in_team, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose)
		human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand()

		# Solve the routing and matching problem to find a plan
		flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, result_dict = global_planner.plan(edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, node_seq, max_iter, flag_initialize, flag_solver)

		# Visualize results
		visualizer = ResultVisualizer()
		visualizer.print_results(route_list, route_time_list, team_list)

		# Save results
		result_dict['flag_success'] = flag_success
		result_dict['route_list'] = route_list
		result_dict['route_time_list'] = route_time_list
		result_dict['team_list'] = team_list
		result_dict['human_in_team'] = human_in_team
		result_dict['y_sol'] = y_sol
		result_dict['z_sol'] = z_sol
		result_dict['veh_num'] = veh_num
		result_dict['node_num'] = node_num
		result_dict['human_choice'] = human_choice
		result_dict['human_num'] = human_num
		result_dict['max_human_in_team'] = max_human_in_team
		result_dict['demand_penalty'] = demand_penalty
		result_dict['time_penalty'] = time_penalty
		result_dict['time_limit'] = time_limit
		result_dict['solver_time_limit'] = solver_time_limit
		result_dict['beta'] = beta
		result_dict['sigma'] = sigma
		result_dict['flag_verbose'] = flag_verbose
		result_dict['edge_time'] = edge_time
		result_dict['node_time'] = node_time
		result_dict['edge_time_std'] = edge_time_std
		result_dict['node_time_std'] = node_time_std
		result_dict['human_demand_bool'] = human_demand_bool
		result_dict['human_demand_int_unique'] = human_demand_int_unique
		result_dict['node_seq'] = node_seq
		result_dict['max_iter'] = max_iter
		result_dict['flag_initialize'] = flag_initialize
		result_dict['flag_solver'] = flag_solver
		result_dict['node_pose'] = node_pose
		# helper.save_dict('result.dat', result_dict)

		robot_1_route = route_list[0]
		robot_2_route = route_list[1]
		robot_3_route = route_list[2]		
		self.final_plan_1 = self.selected_points[robot_1_route]
		self.final_plan_2 = self.selected_points[robot_2_route]
		self.final_plan_3 = self.selected_points[robot_3_route]
		self.final_plan_3d_1 = self.selected_points_3d[robot_1_route]
		self.final_plan_3d_2 = self.selected_points_3d[robot_2_route]
		self.final_plan_3d_3 = self.selected_points_3d[robot_3_route]
		self.publish_3d_plan(self.final_plan_3d_1, 1)
		self.publish_3d_plan(self.final_plan_3d_2, 2)
		self.publish_3d_plan(self.final_plan_3d_3, 3)
		self.publish_plan(self.final_plan_1, 1)
		self.publish_plan(self.final_plan_2, 2)
		self.publish_plan(self.final_plan_3, 3)
		# self.publish_markers()

	def publish_3d_plan(self, plan, robot_number):
		if robot_number==1:
			self._pub_plan_3d_robot_1.publish(np.float32(plan).ravel())
		elif robot_number == 2:
			self._pub_plan_3d_robot_2.publish(np.float32(plan).ravel())
		else:
			self._pub_plan_3d_robot_3.publish(np.float32(plan).ravel())

	def publish_plan(self, plan, robot_number):
		msg = Path()
		msg.header.frame_id = "world"
		msg.header.stamp = rospy.Time.now()
		for wp in plan:
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
		if robot_number == 1:
			self._pub_plan_robot_1.publish(msg)
		elif robot_number == 2:
			self._pub_plan_robot_2.publish(msg)
		else:
			self._pub_plan_robot_3.publish(msg)

	def publish_markers(self):
		msg = MarkerArray()
		counter = 0
		for wp in self.final_plan:
			marker = Marker()
			marker.id = counter
			marker.header.frame_id = "world"
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.SPHERE
			marker.pose.position.x = wp[0]
			marker.pose.position.y = wp[1]
			marker.pose.position.z = 0.0
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.scale.x = 0.2
			marker.scale.y = 0.2
			marker.scale.z = 0.01
			marker.color.a = 1.0
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			counter = counter+1
			msg.markers.append(marker)
		rospy.loginfo("Publishing Markers...")
		self._pub_markers.publish(msg)

	def publish_markers_initial(self):
		msg = MarkerArray()
		counter = 0
		for wp in self.selected_points:
			marker = Marker()
			marker.id = counter
			marker.header.frame_id = "world"
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.SPHERE
			marker.pose.position.x = wp[0]
			marker.pose.position.y = wp[1]
			marker.pose.position.z = 0.0
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.scale.x = 0.7
			marker.scale.y = 0.7
			marker.scale.z = 0.01
			marker.color.a = 1.0
			rgb = get_rgb_from_demand(self.demand_list[counter])
			marker.color.r = 0
			marker.color.g = 0
			marker.color.b = 0
			counter = counter+1
			# msg.markers.append(marker)
			# marker.id = counter
			# marker.type = Marker.TEXT_VIEW_FACING
			# marker.text = str(10)
			# marker.action = Marker.ADD
			# counter = counter+1
			msg.markers.append(marker)
		# for i in range(1,10):
		# 	marker = Marker()
		# 	marker.id = counter;
		# 	marker.header.frame_id = "world"
		# 	marker.header.stamp = rospy.Time.now()
		# 	marker.type = Marker.CUBE
		# 	marker.pose.position.x = 1.78+3.6*(i-1);
		# 	marker.pose.position.y = 18;
		# 	marker.pose.position.z = 0.0;
		# 	marker.pose.orientation.x = 0.0;
		# 	marker.pose.orientation.y = 0.0;
		# 	marker.pose.orientation.z = 0.0;
		# 	marker.pose.orientation.w = 1.0;
		# 	marker.scale.x = 3.6;
		# 	marker.scale.y = 1.4;
		# 	marker.scale.z = 0.01;
		# 	marker.color.a = 1.0; 
		# 	rgb = get_rgb_from_demand(i)
		# 	marker.color.r = rgb[0];
		# 	marker.color.g = rgb[1];
		# 	marker.color.b = rgb[2];
		# 	counter = counter+1
		# 	msg.markers.append(marker)
		rospy.loginfo("Publishing Markers...")
		self._pub_markers_initial.publish(msg)

	def publish_plan_initial(self):
		msg = Path()
		msg.header.frame_id = "world"
		msg.header.stamp = rospy.Time.now()
		for wp in self.selected_points:
			pose = PoseStamped()
			pose.pose.position.x = wp[0]
			pose.pose.position.y = wp[1]
			pose.pose.position.z = 0.0
			pose.pose.orientation.x = 0.0
			pose.pose.orientation.y = 0.0
			pose.pose.orientation.z = 0.0
			pose.pose.orientation.w = 1.0
			msg.poses.append(pose)
		wp = self.selected_points[0]
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
		self._pub_plan_initial.publish(msg)

	
		# self._r.sleep()

	def callback(self,points):
		# self.run_demo()
		self.publish_markers_initial()
		# self.publish_plan_initial()
		# rospy.sleep(10.)
		xy_points = {}
		navigable = []
		xy_points["points"] = self.selected_points
		xy_points["navigable"] = self.navigable
		self.generate_plan()


		# if (tour_plan.selection_done == True):
		# 	print("Already selected points, going to generate plan now!")
		# 	tour_plan.generate_plan()
		# 	return
		# tour_plan.selected_points.append([point.point.x, point.point.y, 0.0])
		# tour_plan.selection_done = bool(input("Want to finish selection?"))
		
	# def run_demo(self):
	# 	# print("Total Demand is" + str(sum(self.demand_list)))
	# 	# for i in range(0,sum(self.demand_list),10):
	# 	# 	self.capacity = i
	# 	# 	print(i)
	# 	# 	self.generate_plan()
	# 	# self.capacity = sum(self.demand_list)
	# 	# self.generate_plan()
	# 	data = self.create_data_model()
	# 	manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
	# 										   data['num_vehicles'], data['depot'])
	# 	routing = pywrapcp.RoutingModel(manager)
	# 	def distance_callback(from_index, to_index):
	# 		"""Returns the distance between the two nodes."""
	# 		# Convert from routing variable Index to distance matrix NodeIndex.
	# 		from_node = manager.IndexToNode(from_index)
	# 		to_node = manager.IndexToNode(to_index)
	# 		return data['distance_matrix'][from_node][to_node]
	# 	dist = 0;
	# 	for i in range(0,22):
	# 		dist+=distance_callback(i,i+1)
	# 		print(i,dist)
	# 	dist+=distance_callback(23,0)
	# 	print(dist)

def main():
	
	env_config_file="configs/tasks/pointnav_rgbd.yaml"
	env = habitat.Env(config=habitat.get_config(env_config_file))
	tour_plan = tour_planner(env)
	# tour_plan.publish_markers_initial()
	# tour_plan.publish_plan_initial()
	rospy.Subscriber("/clicked_point", PointStamped,tour_plan.callback, tour_plan,queue_size=1)
	while not rospy.is_shutdown():
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
