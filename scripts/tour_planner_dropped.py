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
sys.path.insert(1, './tour_planning/')
from MatchRouteWrapper import MatchRouteWrapper
from ResultVisualizer import ResultVisualizer
import helper
from IPython import embed
from nav_msgs.srv import GetPlan
import math
import tf

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
	elif demand <= 3:  # was 5 before for 10 total 
		rgb = [1-(0.25*(demand-1)), 0.25*(demand-1), 0.0]
	else:
		rgb = [0.25*(demand-6), 0.0, 1-(0.25*(demand-6))]
	return rgb	


class tour_planner():
	selection_done = False
	flag_use_vulcan_path = True
	def __init__(self):
		if (not rospy.core.is_initialized()):
			rospy.init_node("get_points",anonymous=False)
		_sensor_rate = 50  # hz
		self._r = rospy.Rate(_sensor_rate)
		self._pub_markers = rospy.Publisher("~points", MarkerArray, queue_size = 1)
		self._pub_plan_initial = rospy.Publisher("~initial_global_plan", Path, queue_size=1)
		self._pub_markers_initial = rospy.Publisher("~selected_points", MarkerArray, queue_size = 1)
		self._pub_plan_3d_robot_1 = rospy.Publisher("robot_1/plan_3d", numpy_msg(Floats),queue_size = 1)
		self._pub_plan_robot_1 = rospy.Publisher("robot_1/global_plan", Path, queue_size=1)
		self.selected_points = []
		self.selected_points_3d = []
		self.final_plan_1 = []
		self.final_plan_3d_1 = []	## Use this to save and then finally display the shortest between nodes instead, TRIBHI!!!!!! 
		self.final_plan_2 = []
		self.final_plan_3d_2 = []	## Use this to save and then finally display the shortest between nodes instead, TRIBHI!!!!!! 
		self.navigable = []
		meters_per_pixel =0.05
		with open('./scripts/all_points.csv', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			for row in spamreader:
				map_points = list(map(float,row))
				self.selected_points.append(list(map(float,row)))			
		# self.demand_list = [0, 1, 1, 3, 6, 3, 6, 8, 8, 1, 2 , 1, 2, 6, 6, 8, 8, 9, 1, 3, 5, 2, 8, 4]
		self.demand_list = [8,6,3,1,6,2,4,1,3,7,9,5,9,5,2,6,6,9,7,5,5,3,2,0,0]
		self.node_num = len(self.selected_points)
		self.capacity = 80
		self.selected_points_pose_stamped = []
		self.path_srv = GetPlan()
		self.vulcan_graph_srv = GetPlan()
		self.get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
		self.get_vulcan_plan = rospy.ServiceProxy('/get_distance', GetPlan)
		print("Created tour planner object")
		self._r.sleep()

	def generate_distance_matrix(self):
		distance_matrix = []
		for start_point in self.selected_points:
			distances = []
			# Check Size of the matrix
			for end_point in self.selected_points:
				euclidean_dist = (start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2
				distances.append(euclidean_dist)
			distance_matrix.append(distances)
		return distance_matrix

	def generate_distance_matrix_pose_stamped(self):
		distance_matrix = []
		for start_point in self.selected_points_pose_stamped:
			distances = []
			for end_point in self.selected_points_pose_stamped:
				self.path_srv.start = start_point
				self.path_srv.goal = end_point
				self.path_srv.tolerance = .5
				path = self.get_plan(self.path_srv.start, self.path_srv.goal, self.path_srv.tolerance)
				prev_x = 0.0
				prev_y = 0.0
				total_distance = 0.0
				first_time = True
				for current_point in path.plan.poses:
					x = current_point.pose.position.x
					y = current_point.pose.position.y
					if not first_time:
						total_distance += math.hypot(prev_x - x, prev_y - y) 
					else:
						first_time = False
					prev_x = x
					prev_y = y
				distances.append(total_distance)
			distance_matrix.append(distances)
		print("iN CREATE MODE, FOUND THE DISTANCE MATRIX ", distance_matrix)
		return distance_matrix

	def generate_distance_vulcan_grid(self):
		print("In Vulcan grid distnace callback")
		distance_matrix = []
		for start_point in self.selected_points_pose_stamped:
			distances = []
			for end_point in self.selected_points_pose_stamped:
				self.vulcan_graph_srv.start = start_point
				self.vulcan_graph_srv.goal = end_point
				self.vulcan_graph_srv.tolerance = .5
				path = self.get_vulcan_plan(self.vulcan_graph_srv.start, self.vulcan_graph_srv.goal, self.vulcan_graph_srv.tolerance)
				# total_total_distance = 0.0
				# for i in range(0,len(path.plan.poses)-1):
				# 	self.path_srv.start = path.plan.poses[i]
				# 	self.path_srv.goal = path.plan.poses[i+1]
				# 	self.path_srv.tolerance = .5
				# 	a_star_path = self.get_plan(self.path_srv.start, self.path_srv.goal, self.path_srv.tolerance)
				# 	prev_x = 0.0
				# 	prev_y = 0.0
				# 	total_distance = 0.0
				# 	first_time = True
				# 	for current_point in a_star_path.plan.poses:
				# 		x = current_point.pose.position.x
				# 		y = current_point.pose.position.y
				# 		if not first_time:
				# 			total_distance += math.hypot(prev_x - x, prev_y - y) 
				# 		else:
				# 			first_time = False
				# 		prev_x = x
				# 		prev_y = y
				# 	total_total_distance = total_total_distance + total_distance
				total_total_distance = path.plan.poses[0].pose.position.x
				distances.append(total_total_distance)
			distance_matrix.append(distances)
		print("iN CREATE MODE, FOUND THE DISTANCE MATRIX ", distance_matrix)
		return distance_matrix

	def create_data_model(self):
		"""Stores the data for the problem."""
		data = {}
		if(flag_use_vulcan_path):
			distance_matrix = self.generate_distance_vulcan_grid()	    
		else:
			distance_matrix = self.generate_distance_matrix_pose_stamped()

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

		veh_num = 1
		node_num = len(self.selected_points)
		demand_penalty = 1000.0
		time_penalty = 10.0
		time_limit = 100 * scale_time
		human_num = 5
		human_choice = 3
		max_iter = 1
		max_human_in_team = np.ones(veh_num, dtype=int) * 10 # (human_num // veh_num + 5)

		node_seq = None
		# node_seq = [[0,1,2], [3,4]]

		angular_offset = 2.9

		flag_load = 0
		if flag_load >= 1:
			# setup_dict = helper.load_dict('setup2.dat')
			# setup_dict = helper.load_dict('setup3_uncertain.dat')
			# setup_dict = helper.load_dict('setup4_guboridet.dat')
			# setup_dict = helper.load_dict('setup6_uncertainty.dat')
			setup_dict = helper.load_dict('sigma/sigma4.dat')
			# setup_dict = helper.load_dict('sigma/sigma4.dat')

		# Initialize spacial maps
		node_pose = self.selected_points
		edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
		veh_speed = 1.25/angular_offset
		edge_time = edge_dist.reshape(node_num,node_num) / veh_speed* scale_time
		node_time = np.ones(node_num) * 5.5 * scale_time
		node_time[-2:] = 0.0
		print('node_time = ', node_time)
		if flag_uncertainty:
		    edge_time_std = edge_time * sigma
		    node_time_std = np.ones((veh_num,node_num), dtype=np.float64) * 1.5
		else:
		    edge_time_std = None
		    node_time_std = None

		# Initialize human selections
		global_planner = MatchRouteWrapper(node_num, human_choice, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose)
		human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand()

		if flag_load >= 1:
			human_demand_bool = setup_dict['human_demand_bool']
			human_demand_int_unique = setup_dict['human_demand_int_unique']
		print('human_demand_bool = \n', human_demand_bool)
		print('human_demand_int_unique = \n', human_demand_int_unique)

		# Do optimization
		if flag_load >= 2:
			flag_solver = 1
		flag_success, route_list, route_time_list, y_sol, result_dict = global_planner.plan(edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, node_seq, max_iter, flag_initialize, flag_solver)
		print('sum_obj in tour planner = demand_penalty * demand_obj + time_penalty * sum_time = %f * %f + %f * %f = %f' % (demand_penalty, result_dict['demand_obj'], time_penalty, result_dict['result_sum_time'], result_dict['sum_obj']))

		if flag_load >= 2:
			route_list = setup_dict['route_list']
			route_time_list = setup_dict['route_time_list']
			team_list = setup_dict['team_list']

		# visualizer = ResultVisualizer()
		# visualizer.print_results(route_list, route_time_list, team_list)
		print('route_list = \n', route_list)

		# Save results
		# result_dict['flag_success'] = flag_success
		# result_dict['route_list'] = route_list
		# result_dict['route_time_list'] = route_time_list
		# result_dict['team_list'] = team_list
		# result_dict['human_in_team'] = human_in_team
		# result_dict['y_sol'] = y_sol
		# result_dict['z_sol'] = z_sol
		# result_dict['veh_num'] = veh_num
		# result_dict['node_num'] = node_num
		# result_dict['human_choice'] = human_choice
		# result_dict['human_num'] = human_num
		# result_dict['max_human_in_team'] = max_human_in_team
		# result_dict['demand_penalty'] = demand_penalty
		# result_dict['time_penalty'] = time_penalty
		# result_dict['time_limit'] = time_limit
		# result_dict['solver_time_limit'] = solver_time_limit
		# result_dict['beta'] = beta
		# result_dict['sigma'] = sigma
		# result_dict['flag_verbose'] = flag_verbose
		# result_dict['edge_time'] = edge_time
		# result_dict['node_time'] = node_time
		# result_dict['edge_time_std'] = edge_time_std
		# result_dict['node_time_std'] = node_time_std
		# result_dict['human_demand_bool'] = human_demand_bool
		# result_dict['human_demand_int_unique'] = human_demand_int_unique
		# result_dict['node_seq'] = node_seq
		# result_dict['max_iter'] = max_iter
		# result_dict['flag_initialize'] = flag_initialize
		# result_dict['flag_solver'] = flag_solver
		# result_dict['node_pose'] = node_pose
		# helper.save_dict('result.dat', result_dict)

		robot_1_route = route_list[0]
		self.final_plan_1 = list(self.selected_points[robot_1_route])
		self.final_plan_pose_stamped = list(np.array(self.selected_points_pose_stamped)[robot_1_route])
		j=0
		if(self.flag_use_vulcan_path):
			for i in range(0,len(self.final_plan_pose_stamped)-1):
				self.vulcan_graph_srv.start = self.final_plan_pose_stamped[i]
				self.vulcan_graph_srv.goal = self.final_plan_pose_stamped[i+1]
				self.vulcan_graph_srv.tolerance = .5
				path = self.get_vulcan_plan(self.vulcan_graph_srv.start, self.vulcan_graph_srv.goal, self.vulcan_graph_srv.tolerance)
				print("Found vulcan path between ", i, " and ", i+1)
				if(len(path.plan.poses)>2):
					k = 0
					for pose in path.plan.poses[1:-1]:
						j = j+1
						k = k+1
						y_diff =  path.plan.poses[k+1].pose.position.y - path.plan.poses[k].pose.position.y
						x_diff =  path.plan.poses[k+1].pose.position.x - path.plan.poses[k].pose.position.x
						yaw = math.atan2(y_diff,x_diff)
						quat = tf.transformations.quaternion_from_euler(0.0,0.0,yaw)
						print(quat)
						node = [pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,quat[0],quat[1],quat[2],quat[3]]						
						self.final_plan_1.insert(i+j,node)
						print(self.final_plan_1)
			print("Number of path points inserted is ", j, "and total is ", len(self.final_plan_1))
		self.publish_plan(self.final_plan_1, 1)
		self.publish_3d_plan(self.final_plan_1,1)
		# self.publish_markers()

	def publish_3d_plan(self, plan, robot_number):
		if robot_number==1:
			self._pub_plan_3d_robot_1.publish(np.float32(plan).ravel())
		elif robot_number == 2:
			self._pub_plan_3d_robot_2.publish(np.float32(plan).ravel())
		else:
			self._pub_plan_3d_robot_3.publish(np.float32(plan).ravel())
			print("Publishing Plan")

	def publish_plan(self, plan, robot_number):
		msg = Path()
		msg.header.frame_id = "map"
		msg.header.stamp = rospy.Time.now()
		for wp in plan:
			pose = PoseStamped()
			pose.pose.position.x = wp[0]
			pose.pose.position.y = wp[1]
			pose.pose.position.z = wp[2]
			pose.pose.orientation.x = wp[3]
			pose.pose.orientation.y = wp[4]
			pose.pose.orientation.z = wp[5]
			pose.pose.orientation.w = wp[6]
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
			marker.id = counter;
			marker.header.frame_id = "map"
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.SPHERE
			marker.pose.position.x = wp[0];
			marker.pose.position.y = wp[1];
			marker.pose.position.z = 0.0;
			marker.pose.orientation.x = 0.0;
			marker.pose.orientation.y = 0.0;
			marker.pose.orientation.z = 0.0;
			marker.pose.orientation.w = 1.0;
			marker.scale.x = 0.2;
			marker.scale.y = 0.2;
			marker.scale.z = 0.01;
			marker.color.a = 1.0; 
			marker.color.r = 1.0;
			marker.color.g = 0.0;
			marker.color.b = 0.0;
			counter = counter+1
			msg.markers.append(marker)
		rospy.loginfo("Publishing Markers...")
		self._pub_markers.publish(msg)

	def publish_markers_initial(self):
		msg = MarkerArray()
		counter = 0
		for wp in self.selected_points:
			marker = Marker()
			marker.id = counter;
			marker.header.frame_id = "map"
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.SPHERE
			marker.pose.position.x = wp[0];
			marker.pose.position.y = wp[1];
			marker.pose.position.z = 0.0;
			marker.pose.orientation.x = 0.0;
			marker.pose.orientation.y = 0.0;
			marker.pose.orientation.z = 0.0;
			marker.pose.orientation.w = 1.0;
			marker.scale.x = 0.7;
			marker.scale.y = 0.7;
			marker.scale.z = 0.01;
			marker.color.a = 1.0; 
			rgb = get_rgb_from_demand(self.demand_list[counter])
			marker.color.r = 0;
			marker.color.g = 0;
			marker.color.b = 0;
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
		# 	marker.header.frame_id = "map"
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
		msg.header.frame_id = "map"
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

	def plan(self, start_point):
		# self.run_demo()
		self.selected_points.append(start_point)
		self.selected_points.append(start_point)
		self.selected_points = np.array(self.selected_points)
		i = 0
		for point in self.selected_points:
			start = PoseStamped()
			start.header.seq = i
			start.header.frame_id = "map"
			start.header.stamp = rospy.Time(0)
			start.pose.position.x = point[0]
			start.pose.position.y = point[1]
			start.pose.position.z = point[2]
			start.pose.orientation.x = point[3]
			start.pose.orientation.y = point[4]
			start.pose.orientation.z = point[5]
			start.pose.orientation.w = point[6]
			self.selected_points_pose_stamped.append(start)
		self.publish_markers_initial()
		# self.generate_plan()
		start_time = rospy.get_time()
		self.generate_distance_vulcan_grid()
		print("Calculated matrix in ", rospy.get_time()-start_time)
		self.generate_distance_matrix_pose_stamped()
		print("Calculated matrix in ", rospy.get_time()-start_time)
		


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
	tour_plan = tour_planner()
	# tour_plan.publish_markers_initial()
	# tour_plan.publish_plan_initial()
	# rospy.Subscriber("/clicked_point", PointStamped,tour_plan.callback, tour_plan,queue_size=1)
	tour_plan.generate_plan()
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
