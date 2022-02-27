import sys
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
import yaml
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import csv
from nav_msgs.srv import GetPlan
import math
import matplotlib.pyplot as plt


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


sys.path.insert(1, './tour_planning/')
from MatchRouteWrapper import MatchRouteWrapper
from ResultVisualizer import ResultVisualizer
from IPython import embed


folder_name = './temp/'

class tour_planner():
    selection_done = False
    selected_points = []
    selected_points_pose_stamped = []
    flag_use_vulcan_path = False
    def __init__(self):
        if (not rospy.core.is_initialized()):
            rospy.init_node("get_points",anonymous=False)
        _sensor_rate = 50  # hz
        self._r = rospy.Rate(_sensor_rate)
        self._pub_markers_initial = rospy.Publisher("~selected_points", MarkerArray, queue_size = 1)
        self._pub_plan_3d_robot_1 = rospy.Publisher("robot_1/plan_3d", numpy_msg(Floats),queue_size = 1)
        self._pub_plan_robot_1 = rospy.Publisher("robot_1/global_plan", Path, queue_size=1)
        depot = [0.0,0.0,0.0,0.0,0.0,0.0,1.0]
        with open('./scripts/all_points.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                map_points = list(map(float,row))
                self.selected_points.append(list(map(float,row)))
        self.selected_points.append(depot)
        self.selected_points.append(depot)
        for point in self.selected_points:
            start = PoseStamped()
            start.header.seq = 1
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
        self.path_srv = GetPlan()
        self.vulcan_graph_srv = GetPlan()
        self.get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        self.get_vulcan_plan = rospy.ServiceProxy('/get_distance', GetPlan)
        print("Created tour planner object")
        self._r.sleep()

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
        print("In Vulcan grid distance callback")
        distance_matrix = []
        for start_point in self.selected_points_pose_stamped[0:1]:
            distances = []
            for end_point in self.selected_points_pose_stamped[0:1]:
                self.vulcan_graph_srv.start = start_point
                self.vulcan_graph_srv.goal = end_point
                self.vulcan_graph_srv.tolerance = .5
                path = self.get_vulcan_plan(self.vulcan_graph_srv.start, self.vulcan_graph_srv.goal, self.vulcan_graph_srv.tolerance)
                total_total_distance = 0.0
                for i in range(0,len(path.plan.poses)-1):
                    self.path_srv.start = path.plan.poses[i]
                    self.path_srv.goal = path.plan.poses[i+1]
                    self.path_srv.tolerance = .5
                    print("before calling the service")
                    a_star_path = self.get_plan(self.path_srv.start, self.path_srv.goal, self.path_srv.tolerance)
                    print(a_star_path.plan.poses.size())
                    # prev_x = 0.0
                    # prev_y = 0.0
                    # total_distance = 0.0
                    # first_time = True
                    # for current_point in a_star_path.plan.poses:
                    # 	x = current_point.pose.position.x
                    # 	y = current_point.pose.position.y
                    # 	if not first_time:
                    # 		total_distance += math.hypot(prev_x - x, prev_y - y) 
                    # 	else:
                    # 		first_time = False
                    # 	prev_x = x
                    # 	prev_y = y
                    # total_total_distance = total_total_distance + total_distance
                total_total_distance = path.plan.poses[0].pose.position.x
                distances.append(total_total_distance)
            distance_matrix.append(distances)
        print("iN CREATE MODE, FOUND THE DISTANCE MATRIX ", distance_matrix)
        return distance_matrix
    
    def create_data_model(self):
        """Stores the data for the problem."""
        data = {}
        if(self.flag_use_vulcan_path):
            distance_matrix = self.generate_distance_vulcan_grid()	    
        else:
            distance_matrix = self.generate_distance_matrix_pose_stamped()
        data['distance_matrix'] = distance_matrix  # yapf: disable
        # data['demands'] = self.demand_list
        # data['vehicle_capacities'] = [self.capacity]
        # data['depot'] = 0
        return data	

    def generate_plan(self, time_limit = 1000 , std_dev = 3, human_num = 5, human_choice = 5):
        data = self.create_data_model()
        flag_verbose = True

        flag_initialize = 0 # 0: VRP, 1: random
        flag_solver = 1 # 0 GUROBI exact solver, 1: OrTool heuristic solver
        solver_time_limit = 300.0
        flag_uncertainty = False
        beta = 0.98
        sigma = 1.0
        flag_load = 0.0
        scale_time = 1.0

        veh_num = 1
        node_num = len(self.selected_points)
        demand_penalty = 1000.0
        time_penalty = 10.0
        # time_limit = 1000 * scale_time
        # human_num = 5
        # human_choice = 5
        max_iter = 1
        max_human_in_team = np.ones(veh_num, dtype=int) * 10 # (human_num // veh_num + 5)

        node_seq = None
        # node_seq = [[0,1,2], [3,4]]

        angular_offset = 2.9


        # Initialize spacial maps
        node_pose = self.selected_points
        edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
        veh_speed = 1.25/angular_offset
        veh_speed = veh_speed/3
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
        human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(std_dev)

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

    
        # helper.save_dict('result.csv', result_dict)

        robot_1_route = route_list[0]
        print("Route time is ", route_time_list)
        # self.final_plan_1 = list(self.selected_points[robot_1_route])
        # self.final_plan_pose_stamped = list(np.array(self.selected_points_pose_stamped)[robot_1_route])
        return result_dict

def main():
    tour_plan = tour_planner()
    full_result_list = []
    full_result = {}
    for i in range(1,10,2):
        for j in range(1,30,5):
            human_num = i
            human_choice = 5
            std_dev = j
            result_dict = tour_plan.generate_plan(time_limit = 1000, std_dev = std_dev, human_choice = human_choice, human_num = human_num)
            actual_time_limit = 200
            if (i ==1):   
                full_result['std_dev'] = [std_dev]
                full_result['human_num'] = [human_num]
                full_result['human_choice'] = [human_choice]
                full_result['time_limit'] = [actual_time_limit]
                full_result['sum_obj'] = [result_dict['sum_obj']]
                full_result['total_demand'] = [result_dict['total_demand']]
                full_result['demand_obj'] = [result_dict['demand_obj']]
                full_result['dropped_demand_rate'] = [result_dict['dropped_demand_rate']]
                full_result['result_max_time'] = [result_dict['result_max_time']]
                full_result['optimization_time'] = [result_dict['optimization_time']]
                full_result['success'] = [result_dict['success']]       
            else:
                full_result['std_dev'].append(i)
                full_result['human_num'].append(human_num)
                full_result['human_choice'].append(human_choice)
                full_result['time_limit'].append(actual_time_limit)
                full_result['sum_obj'].append(result_dict['sum_obj'])
                full_result['total_demand'].append(result_dict['total_demand'])
                full_result['demand_obj'].append(result_dict['demand_obj'])
                full_result['dropped_demand_rate'].append(result_dict['dropped_demand_rate'])
                full_result['result_max_time'].append(result_dict['result_max_time'])
                full_result['optimization_time'].append(result_dict['optimization_time'])
                full_result['success'].append(result_dict['success'])
    print(full_result)

    visualizer = ResultVisualizer()

    visualizer.save_plots(folder_name)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(full_result['human_num'], full_result['result_max_time'])
    ax.set_xlabel('human_num')
    ax.set_ylabel('Tour Plan time')
    fig_file = folder_name + "plan_time_w_human_num.png"
    fig.savefig(fig_file, bbox_inches='tight')

    field_names = list(full_result.keys())
    with open('human_num_with_std_bins.csv','w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames = field_names, extrasaction='ignore', delimiter = ';')
        writer.writeheader()
        writer.writerows(full_result)
    
    embed()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == "__main__":
    main()
