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

def save_results(full_result, file_appender):
        visualizer = ResultVisualizer()

        # visualizer.save_plots(folder_name)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(full_result['total_demand'], full_result['dropped_demand_rate'])
        # ax.set_xlabel('total_demand')
        # ax.set_ylabel('dropped_demand_rate')
        # fig_file = folder_name + "obj" + file_appender +".png"
        # fig.savefig(fig_file, bbox_inches='tight')
        csv_file = folder_name + "batch_20_demand_plots_time_800_" + file_appender + ".csv"
        keys = sorted(full_result.keys())
        with open(csv_file,'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ",")
            writer.writerow(keys)
            writer.writerows(zip(*[full_result[key] for key in keys]))
        

def get_rgb_from_demand(demand):
	rgb = []
	if demand == 0:
		rgb = [0.0,0.0,0.0]
	elif demand <= 5:  # was 5 before for 10 total 
		rgb = [1-(0.25*(demand-1)), 0.25*(demand-1), 0.0]
	else:
		rgb = [0.25*(demand-6), 0.0, 1-(0.25*(demand-6))]
	return rgb	

class tour_planner():
    selection_done = False
    selected_points = []
    selected_points_pose_stamped = []
    flag_use_vulcan_path = False
    publissh_plans = True
    def __init__(self):
        if (not rospy.core.is_initialized()):
            rospy.init_node("get_points",anonymous=False)
        _sensor_rate = 50  # hz
        self._r = rospy.Rate(_sensor_rate)
        self._pub_markers_initial = rospy.Publisher("~selected_points", MarkerArray, queue_size = 1)
        self._pub_plan_3d_robot_1 = rospy.Publisher("robot_1/plan_3d", numpy_msg(Floats),queue_size = 1)
        self._pub_plan_robot_1 = rospy.Publisher("robot_1/global_plan", Path, queue_size=1)
        self._pub_plan_robot_2 = rospy.Publisher("robot_2/global_plan", Path, queue_size=1)
        self._pub_plan_robot_3 = rospy.Publisher("robot_3/global_plan", Path, queue_size=1)
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
        self.selected_points = np.array(self.selected_points)
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
        # print("iN CREATE MODE, FOUND THE DISTANCE MATRIX ", distance_matrix)
        return distance_matrix

    def generate_distance_vulcan_grid(self):
        print("In Vulcan grid distance callback")
        distance_matrix = []
        for start_point in self.selected_points_pose_stamped:
            distances = []
            for end_point in self.selected_points_pose_stamped:
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
        # print("iN CREATE MODE, FOUND THE DISTANCE MATRIX ", distance_matrix)
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

    def generate_plan(self, time_limit = 1000 , std_dev = 3, human_num = 5, human_choice = 5, human_demand_bool=[], human_demand_int_unique=[]):
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
        max_iter = 1
        max_human_in_team = np.ones(veh_num, dtype=int) * 10 # (human_num // veh_num + 5)

        node_seq = None
        # node_seq = [[0,15], [3,4,13]]

        angular_offset = 2.9


        # Initialize spacial maps
        node_pose = self.selected_points
        edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
        veh_speed = 1.25/angular_offset
        veh_speed = veh_speed/3
        edge_time = edge_dist.reshape(node_num,node_num) / veh_speed* scale_time
        node_time = np.ones(node_num) * 5.5 * scale_time
        node_time[-2:] = 0.0
        # print('node_time = ', node_time)
        if flag_uncertainty:
            edge_time_std = edge_time * sigma
            node_time_std = np.ones((veh_num,node_num), dtype=np.float64) * 1.5
        else:
            edge_time_std = None
            node_time_std = None

        # # Initialize human selections
        global_planner = MatchRouteWrapper(node_num, human_choice, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose)
        # human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(std_dev)

        if flag_load >= 1:
            human_demand_bool = setup_dict['human_demand_bool']
            human_demand_int_unique = setup_dict['human_demand_int_unique']
        # print('human_demand_bool = \n', human_demand_bool)
        # print('human_demand_int_unique = \n', human_demand_int_unique)

        # Do optimization
        if flag_load >= 2:
            flag_solver = 1
        flag_success, route_list, route_time_list, y_sol, result_dict = global_planner.plan(edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, node_seq, max_iter, flag_initialize, flag_solver)
        # print('sum_obj in tour planner = demand_penalty * demand_obj + time_penalty * sum_time = %f * %f + %f * %f = %f' % (demand_penalty, result_dict['demand_obj'], time_penalty, result_dict['result_sum_time'], result_dict['sum_obj']))

        if flag_load >= 2:
            route_list = setup_dict['route_list']
            route_time_list = setup_dict['route_time_list']
            team_list = setup_dict['team_list']

        # visualizer = ResultVisualizer()
        # visualizer.print_results(route_list, route_time_list, team_list)
        # print('route_list = \n', route_list)

    
        # helper.save_dict('result.csv', result_dict)

        robot_1_route = route_list
        # print("Route time is ", route_time_list)
        self.final_plan_1 = list(self.selected_points[robot_1_route])
        self.final_plan_pose_stamped = list(np.array(self.selected_points_pose_stamped)[robot_1_route])
        return result_dict

    def greedy_agent(self, time_limit = 1000 , std_dev = 3, human_num = 5, human_choice = 5, flag_greedy_in_time = True, human_demand_bool = [], human_demand_int_unique= []):
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
        max_iter = 1
        max_human_in_team = np.ones(veh_num, dtype=int) * 10 # (human_num // veh_num + 5)

        node_seq = None
        node_seq = [[0,15], [3,4,13]]

        angular_offset = 2.9


        # Initialize spacial maps
        node_pose = self.selected_points
        edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
        veh_speed = 1.25/angular_offset
        veh_speed = veh_speed/3
        edge_time = edge_dist.reshape(node_num,node_num) / veh_speed* scale_time
        node_time = np.ones(node_num) * 5.5 * scale_time
        node_time[-2:] = 0.0
        # print('node_time = ', node_time)
        if flag_uncertainty:
            edge_time_std = edge_time * sigma
            node_time_std = np.ones((veh_num,node_num), dtype=np.float64) * 1.5
        else:
            edge_time_std = None
            node_time_std = None
        start_time = rospy.get_time()
        # Initialize human selections
        global_planner = MatchRouteWrapper(node_num, human_choice, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose)
        # human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(std_dev)

        if flag_load >= 1:
            human_demand_bool = setup_dict['human_demand_bool']
            human_demand_int_unique = setup_dict['human_demand_int_unique']
        # print('human_demand_bool = \n', human_demand_bool)
        # print('human_demand_int_unique = \n', human_demand_int_unique)

        place_num = node_num-2
        penalty_mat = np.zeros(place_num, dtype=np.float64) # (veh_num, place_num)
        for i in range(place_num):
            penalty_mat[i] = (human_demand_bool[:,i]).sum()


        if(flag_greedy_in_time):
            demanded_pois = []
            for i in range(place_num):
                if (penalty_mat[i]>0):
                    demanded_pois.append(i)
            route_list = []
            current_poi_index = place_num
            route_list.append(current_poi_index)
            max_time_to_get_back = max(edge_time[demanded_pois,-2]) + node_time[5]
            total_time = 0.0
            route_time_list = []
            route_time_list.append(total_time)
            while ((total_time <= time_limit-max_time_to_get_back) and (len(demanded_pois)>0)): 
                next_poi_index = demanded_pois[np.argmin(edge_time[current_poi_index, demanded_pois])]
                
                total_time = total_time+ edge_time[current_poi_index, next_poi_index] + node_time[current_poi_index]
                if (total_time >=time_limit-max_time_to_get_back):
                    total_time = total_time - edge_time[current_poi_index, next_poi_index] - node_time[current_poi_index]
                    break
                demanded_pois.remove(next_poi_index)
                route_list.append(next_poi_index)
                route_time_list.append(total_time)
                current_poi_index = next_poi_index
            route_list.append(place_num)
            total_time = total_time+ edge_time[current_poi_index, place_num] + node_time[current_poi_index]
            route_time_list.append(total_time)
            if (total_time>time_limit):
                print("time limit exceeded")
                embed()
            y_sol = np.zeros(place_num)
            y_sol[route_list[1:-1]] = 1.0
            demand_satisfied = penalty_mat[route_list[1:-1]].sum()
            sum_obj, demand_obj, result_sum_time,  node_visit = global_planner.evaluator.objective_fcn(edge_time, node_time, [route_list], y_sol, human_demand_bool)
            result_dict = {}
            result_dict['sum_obj'] = sum_obj
            result_dict['total_demand'] = penalty_mat.sum()
            result_dict['demand_obj'] = demand_obj
            result_dict['dropped_demand_rate'] = result_dict['demand_obj']/result_dict['total_demand']
            result_dict['result_max_time'] = route_time_list[-1]
            result_dict['optimization_time'] = rospy.get_time()-start_time
            result_dict['success'] = True
            
            robot_1_route = route_list
            # print("Route time is ", route_time_list)
            self.final_plan_1 = list(self.selected_points[robot_1_route])
            self.final_plan_pose_stamped = list(np.array(self.selected_points_pose_stamped)[robot_1_route])
            return result_dict
        else:
            demanded_pois = []
            print(human_demand_int_unique)
            demands = list(np.nonzero(penalty_mat)[0])
            unsorted_demanded_pois = penalty_mat[demands]
            frequency_order = list(np.argsort(unsorted_demanded_pois))
            frequency_order.reverse()
            demanded_pois = np.array(demands)
            demanded_pois = list(demanded_pois[frequency_order]) 
            route_list = []
            current_poi_index = place_num
            route_list.append(current_poi_index)
            time_to_get_back_to_depot = edge_time[demanded_pois[-1],-2] + node_time[5]
            total_time = 0.0
            route_time_list = []
            route_time_list.append(total_time)
            i = 0
            # if(penalty_mat.sum()==1):
            #     embed()
            print("Demanded POIs are", demanded_pois)
            while ((total_time <= time_limit-time_to_get_back_to_depot) and (len(demanded_pois)>0)): 
                next_poi_index = demanded_pois[i]
                total_time = total_time+ edge_time[current_poi_index, next_poi_index] + node_time[current_poi_index]
                time_to_get_back_to_depot = edge_time[next_poi_index,-2] + node_time[next_poi_index]
                if (total_time >=time_limit-time_to_get_back_to_depot):
                    total_time = total_time - edge_time[current_poi_index, next_poi_index] - node_time[current_poi_index]
                    break
                route_list.append(next_poi_index)
                route_time_list.append(total_time)
                demanded_pois.remove(next_poi_index)
                current_poi_index = next_poi_index
            route_list.append(place_num)
            print("Greedy in demand route list", route_list)
            total_time = total_time+ edge_time[current_poi_index, place_num] + node_time[current_poi_index]
            route_time_list.append(total_time)
            if (total_time>time_limit):
                print("time limit exceeded")
                embed()
            y_sol = np.zeros(place_num)
            y_sol[route_list[1:-1]] = 1.0
            demand_satisfied = penalty_mat[route_list[1:-1]].sum()
            sum_obj, demand_obj, result_sum_time,  node_visit = global_planner.evaluator.objective_fcn(edge_time, node_time, [route_list], y_sol, human_demand_bool)
            result_dict = {}
            result_dict['sum_obj'] = sum_obj
            result_dict['total_demand'] = penalty_mat.sum()
            result_dict['demand_obj'] = demand_obj
            result_dict['dropped_demand_rate'] = result_dict['demand_obj']/result_dict['total_demand']
            result_dict['result_max_time'] = route_time_list[-1]
            result_dict['optimization_time'] = rospy.get_time()-start_time
            result_dict['success'] = True
            robot_1_route = route_list
            # print("Route time is ", route_time_list)
            self.final_plan_1 = list(self.selected_points[robot_1_route])
            self.final_plan_pose_stamped = list(np.array(self.selected_points_pose_stamped)[robot_1_route])
            return result_dict
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
            


def main():
    tour_plan = tour_planner()
    full_result_list = []
    full_result_greedy = {}
    full_result = {}
    full_result_greedy_demand = {}
    flag_visualize = False
    flag_initialize = 0 # 0: VRP, 1: random
    flag_solver = 1 # 0 GUROBI exact solver, 1: OrTool heuristic solver
    solver_time_limit = 300.0
    flag_uncertainty = False
    beta = 0.98
    sigma = 1.0
    flag_load = 0.0
    scale_time = 1.0

    veh_num = 1
    node_num = len(tour_plan.selected_points)
    demand_penalty = 1000.0
    time_penalty = 10.0
    max_iter = 1
    max_human_in_team = np.ones(veh_num, dtype=int) * 10 # (human_num // veh_num + 5)

    node_seq = None
    node_seq = [[0,15], [3,4,13]]

    angular_offset = 2.9
    data = tour_plan.create_data_model()

    # Initialize spacial maps
    node_pose = tour_plan.selected_points
    edge_dist = np.array(data['distance_matrix']) # squareform(pdist(node_pose))
    veh_speed = 1.25/angular_offset
    veh_speed = veh_speed/3
    edge_time = edge_dist.reshape(node_num,node_num) / veh_speed* scale_time
    node_time = np.ones(node_num) * 5.5 * scale_time
    node_time[-2:] = 0.0
    # print('node_time = ', node_time)
    if flag_uncertainty:
        edge_time_std = edge_time * sigma
        node_time_std = np.ones((veh_num,node_num), dtype=np.float64) * 1.5
    else:
        edge_time_std = None
        node_time_std = None
    start_time = rospy.get_time()
    human_num = 1
    human_choice = 5
    std_dev = 5
    time_limit = 300
    global_planner = MatchRouteWrapper(node_num, human_choice, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose = False)
    human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(std_dev)
    place_num = node_num-2
    penalty_mat = np.zeros(place_num, dtype=np.float64) # (veh_num, place_num)
    for k in range(place_num):
        penalty_mat[k] = (human_demand_bool[:,k]).sum()
    total_demand_req = penalty_mat.sum()
    # Initialize human selections
    
    total_demand_list = [5,10,15,20,25,30]
    number_of_samples = 20*np.ones(len(total_demand_list))
    got_demand_right = np.array(np.zeros(len(total_demand_list)), dtype = bool)
    counter_of_samples = np.zeros(len(total_demand_list))
    while(not np.all(got_demand_right)):
        for i in range(1,10):
            # for j in range(5,20,5):
            human_num = i
            human_choice = 5
            std_dev = 1
            time_limit = 800
            global_planner = MatchRouteWrapper(node_num, human_choice, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose = False)
            human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(std_dev = std_dev)
            penalty_mat = np.zeros(place_num, dtype=np.float64) # (veh_num, place_num)
            for k in range(place_num):
                penalty_mat[k] = (human_demand_bool[:,k]).sum()
            total_demand_req = int(penalty_mat.sum())
            if (total_demand_req not in total_demand_list):
                continue
            index = total_demand_list.index(total_demand_req)
            counter_of_samples[index] = counter_of_samples[index] + 1
            got_demand_right = counter_of_samples > number_of_samples
            if (got_demand_right[index]):
                continue
            # got_demand_right[total_demand_req-1] = True
            result_dict = tour_plan.greedy_agent(time_limit = time_limit, std_dev = std_dev, human_choice = human_choice, human_num = human_num, flag_greedy_in_time = True, human_demand_bool = human_demand_bool, human_demand_int_unique = human_demand_int_unique)
            if (len(full_result_greedy)==0):   
                full_result_greedy['std_dev'] = [std_dev]
                full_result_greedy['human_num'] = [human_num]
                full_result_greedy['human_choice'] = [human_choice]
                full_result_greedy['time_limit'] = [time_limit]
                full_result_greedy['sum_obj'] = [result_dict['sum_obj']]
                full_result_greedy['total_demand'] = [result_dict['total_demand']]
                full_result_greedy['demand_obj'] = [result_dict['demand_obj']]
                full_result_greedy['dropped_demand_rate'] = [result_dict['dropped_demand_rate']]
                full_result_greedy['result_max_time'] = [result_dict['result_max_time']]
                full_result_greedy['optimization_time'] = [result_dict['optimization_time']]
                full_result_greedy['success'] = [result_dict['success']]       
            else:
                full_result_greedy['std_dev'].append(std_dev)
                full_result_greedy['human_num'].append(human_num)
                full_result_greedy['human_choice'].append(human_choice)
                full_result_greedy['time_limit'].append(time_limit)
                full_result_greedy['sum_obj'].append(result_dict['sum_obj'])
                full_result_greedy['total_demand'].append(result_dict['total_demand'])
                full_result_greedy['demand_obj'].append(result_dict['demand_obj'])
                full_result_greedy['dropped_demand_rate'].append(result_dict['dropped_demand_rate'])
                full_result_greedy['result_max_time'].append(result_dict['result_max_time'])
                full_result_greedy['optimization_time'].append(result_dict['optimization_time'])
                full_result_greedy['success'].append(result_dict['success'])
            result_dict = {}
            result_dict = tour_plan.greedy_agent(time_limit = time_limit, std_dev = std_dev, human_choice = human_choice, human_num = human_num, flag_greedy_in_time = False, human_demand_bool = human_demand_bool, human_demand_int_unique = human_demand_int_unique)
            if (len(full_result_greedy_demand)==0):   
                full_result_greedy_demand['std_dev'] = [std_dev]
                full_result_greedy_demand['human_num'] = [human_num]
                full_result_greedy_demand['human_choice'] = [human_choice]
                full_result_greedy_demand['time_limit'] = [time_limit]
                full_result_greedy_demand['sum_obj'] = [result_dict['sum_obj']]
                full_result_greedy_demand['total_demand'] = [result_dict['total_demand']]
                full_result_greedy_demand['demand_obj'] = [result_dict['demand_obj']]
                full_result_greedy_demand['dropped_demand_rate'] = [result_dict['dropped_demand_rate']]
                full_result_greedy_demand['result_max_time'] = [result_dict['result_max_time']]
                full_result_greedy_demand['optimization_time'] = [result_dict['optimization_time']]
                full_result_greedy_demand['success'] = [result_dict['success']]       
            else:
                full_result_greedy_demand['std_dev'].append(std_dev)
                full_result_greedy_demand['human_num'].append(human_num)
                full_result_greedy_demand['human_choice'].append(human_choice)
                full_result_greedy_demand['time_limit'].append(time_limit)
                full_result_greedy_demand['sum_obj'].append(result_dict['sum_obj'])
                full_result_greedy_demand['total_demand'].append(result_dict['total_demand'])
                full_result_greedy_demand['demand_obj'].append(result_dict['demand_obj'])
                full_result_greedy_demand['dropped_demand_rate'].append(result_dict['dropped_demand_rate'])
                full_result_greedy_demand['result_max_time'].append(result_dict['result_max_time'])
                full_result_greedy_demand['optimization_time'].append(result_dict['optimization_time'])
                full_result_greedy_demand['success'].append(result_dict['success'])
            result_dict = {}
            result_dict = tour_plan.generate_plan(time_limit = time_limit, std_dev = std_dev, human_choice = human_choice, human_num = human_num, human_demand_bool = human_demand_bool, human_demand_int_unique = human_demand_int_unique)
            if (len(full_result)==0):   
                full_result['std_dev'] = [std_dev]
                full_result['human_num'] = [human_num]
                full_result['human_choice'] = [human_choice]
                full_result['time_limit'] = [time_limit]
                full_result['sum_obj'] = [result_dict['sum_obj']]
                full_result['total_demand'] = [result_dict['total_demand']]
                full_result['demand_obj'] = [result_dict['demand_obj']]
                full_result['dropped_demand_rate'] = [result_dict['dropped_demand_rate']]
                full_result['result_max_time'] = [result_dict['result_max_time']]
                full_result['optimization_time'] = [result_dict['optimization_time']]
                full_result['success'] = [result_dict['success']]       
            else:
                full_result['std_dev'].append(std_dev)
                full_result['human_num'].append(human_num)
                full_result['human_choice'].append(human_choice)
                full_result['time_limit'].append(time_limit)
                full_result['sum_obj'].append(result_dict['sum_obj'])
                full_result['total_demand'].append(result_dict['total_demand'])
                full_result['demand_obj'].append(result_dict['demand_obj'])
                full_result['dropped_demand_rate'].append(result_dict['dropped_demand_rate'])
                full_result['result_max_time'].append(result_dict['result_max_time'])
                full_result['optimization_time'].append(result_dict['optimization_time'])
                full_result['success'].append(result_dict['success'])
            print("Here we go ", got_demand_right)
    print(full_result)

    save_results(full_result_greedy, "std_dev_1_greedy_in_time")
    save_results(full_result_greedy_demand, "std_dev_1_greedy_in_demand")
    save_results(full_result, "std_dev_1_optimal")

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == "__main__":
    main()
