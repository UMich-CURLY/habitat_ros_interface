import numpy as np
from numpy.lib.function_base import place
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

class OrtoolRoutingSolver:
    def __init__(self, node_num, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit = 20):
        self.LARGETIME = 1000.0                 # A large number that should be larger than any possible time appeared in the optimization problem
        self.node_num = node_num                # The number of nodes
        self.human_num = human_num              # The number of human
        self.demand_penalty = demand_penalty    # The penalty on dropping a POI
        self.time_penalty = time_penalty        # The penalty on the total time consumption of the tour
        if time_limit <= 1:                     # The time limit of the tours, integer valued
            self.time_limit = 300000
        else:
            self.time_limit = int(time_limit)

        self.start_node = self.node_num - 2     # Keep it, the start node is assumed to be No. node_num-2
        self.global_penalty = 1.0               # Keep this constant
        self.solver_time_limit = int(solver_time_limit)     # Time limit for the solver

    def optimize_sub(self, edge_time, node_time, human_demand_bool, node_seq, route_list = None, flag_verbose = False):
        '''
        Optimize the routing problem
        ------------------------------------------------------
        z_sol:             (human_num, veh_num)
        human_demand_bool: (human_num, place_num), i.e. (human_num, node_num - 2)
        '''
        place_num = self.node_num-2
        penalty_mat = np.zeros(place_num, dtype=np.float64)

        result_dict = {}
        result_dict['Optimized'] = True
        result_dict['Status'] = []
        start_time = time.time()

        # Create sub-routing model
        self.sub_solution = []
        self.sub_manager = pywrapcp.RoutingIndexManager(self.node_num-1, 1, self.start_node)
        self.sub_solver = pywrapcp.RoutingModel(self.sub_manager)
        self.sub_solution.append(None)
        for i in range(place_num):
            penalty_mat[i] = human_demand_bool[0][i].sum()
        # print('penalty_mat = ', penalty_mat)
        def temp_distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self.sub_manager.IndexToNode(from_index)
            to_node = self.sub_manager.IndexToNode(to_index)
            if from_node == to_node:
                dist_out = 0.0
            else:
                dist_out = edge_time[from_node,to_node] + node_time[from_node]
            return dist_out

        transit_callback_index = self.sub_solver.RegisterTransitCallback(temp_distance_callback)

        # Define cost of each arc.
        self.sub_solver.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Time'
        self.sub_solver.AddDimension(
            transit_callback_index,
            0,  # no slack
            self.time_limit,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        # distance_dimension = self.sub_solver[k].GetDimensionOrDie(dimension_name)
        # temp_penalty = int(self.global_penalty * self.time_penalty)
        # distance_dimension.SetGlobalSpanCostCoefficient(temp_penalty)

        # Allow to drop nodes.
        for i in range(place_num):
            # temp_penalty = int(penalty_mat[k, i] * self.demand_penalty)
            temp_penalty = int(penalty_mat[i] * self.demand_penalty * self.global_penalty)
            self.sub_solver.AddDisjunction([self.sub_manager.NodeToIndex(i)], temp_penalty)

        # Add sequence constraints, see the references in README.md
        if node_seq is not None:
            self.add_seq_constraint(self.sub_solver, self.sub_manager, node_seq)

        # Solve the problem.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.solver_time_limit
        if (route_list is not None) and len(route_list) > 2:
            initial_solution = self.sub_solver.ReadAssignmentFromRoutes([route_list[1:-1]], True)
            a_sub_solution = self.sub_solver.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
        else:
            search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            a_sub_solution = self.sub_solver.SolveWithParameters(search_parameters)

        # Construct the result dictionary
        result_dict['Status'].append(self.sub_solver.status())
        if self.sub_solver.status() != 1:
            result_dict['Optimized'] = False
        self.sub_solution = a_sub_solution

        end_time = time.time()
        result_dict['Runtime'] = end_time - start_time

        # result_dict['IterCount'] = self.solver.iterations()
        # result_dict['NodeCount'] = self.solver.nodes()
        if flag_verbose:
            print('Solution found: %d' % result_dict['Optimized'])
            print('Optimization status:', result_dict['Status'])
            print('Problem solved in %f seconds' % result_dict['Runtime'])
        flag_success = result_dict['Optimized']
        return flag_success, result_dict

    def set_model(self, edge_time, node_time, node_seq = None):
        # Create Routing Model.
        self.manager = pywrapcp.RoutingIndexManager(self.node_num-1, 1,self.start_node)
        self.solver = pywrapcp.RoutingModel(self.manager)
        self.solution = None

        distance_matrix = edge_time[:self.node_num-1, :self.node_num-1] + 0
        distance_matrix += node_time[:self.node_num-1].reshape(self.node_num-1, 1)

        self.data = {}
        self.data['edge_time'] = edge_time
        self.data['node_time'] = node_time
        self.data['distance_matrix'] = distance_matrix

        transit_callback_index = self.solver.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc.
        self.solver.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Time'
        self.solver.AddDimension(
            transit_callback_index,
            0,  # no slack
            300000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = self.solver.GetDimensionOrDie(dimension_name)
        temp_penalty = int(self.global_penalty * self.time_penalty)
        distance_dimension.SetGlobalSpanCostCoefficient(temp_penalty)

    def add_seq_constraint(self, solver, manager, node_seq):
        distance_dimension = solver.GetDimensionOrDie('Time')
        for i_seq in range(len(node_seq)):
            for i_node in range(len(node_seq[i_seq]) - 1):
                node_i = node_seq[i_seq][i_node]
                node_j = node_seq[i_seq][i_node+1]
                nodeid_i = manager.NodeToIndex(node_i)
                nodeid_j = manager.NodeToIndex(node_j)
                # print('node:', node_i, node_j, 'index:', nodeid_i, nodeid_j)
                # solver.AddPickupAndDelivery(nodeid_i, nodeid_j)
                # solver.solver().Add(solver.VehicleVar(nodeid_i) == solver.VehicleVar(nodeid_j))
                # solver.solver().Add(distance_dimension.CumulVar(nodeid_i) <= distance_dimension.CumulVar(nodeid_j))

                # j active is based on i active
                solver.solver().Add(solver.ActiveVar(nodeid_j) <= solver.ActiveVar(nodeid_i))
                # j's time is after i's time (visit i before j)
                solver.solver().Add(distance_dimension.CumulVar(nodeid_i) <= distance_dimension.CumulVar(nodeid_j))
                # i and j should be using the same vehicle
                constraintActive = solver.ActiveVar(nodeid_i) * solver.ActiveVar(nodeid_j)
                solver.solver().Add(constraintActive * (solver.VehicleVar(nodeid_i) - solver.VehicleVar(nodeid_j)) == 0 )

    def optimize(self, flag_verbose = False):
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.solver_time_limit
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        start_time = time.time()
        self.solution = self.solver.SolveWithParameters(search_parameters)
        end_time = time.time()

        # Construct the result dictionary
        result_dict = {}
        result_dict['Optimized'] = self.solver.status() == 1
        result_dict['Status'] = self.solver.status()
        result_dict['Runtime'] = end_time - start_time
        # result_dict['IterCount'] = self.solver.iterations()
        # result_dict['NodeCount'] = self.solver.nodes()

        # Print the results
        if flag_verbose:
            print('Solution found: %d' % result_dict['Optimized'])
            print('Optimization status: %d' % result_dict['Status'])
            print('Problem solved in %f seconds' % result_dict['Runtime'])
            # print('Problem solved in %d iterations' % result_dict['IterCount'])
            # print('Problem solved in %d branch-and-bound nodes' % result_dict['NodeCount'])
        return result_dict

    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        if from_node == to_node:
            dist_out = 0.0
        else:
            # dist_out = self.data['distance_matrix'][from_node][to_node]
            dist_out = self.data['edge_time'][from_node,to_node] + self.data['node_time'][from_node]
        # print('dist_out = ', dist_out)
        return dist_out

    def get_random_plan(self, edge_time, node_time):
        # Initialize a random routing plan
        place_num = self.node_num-2
        place_perm = np.random.permutation(place_num)
        route_node_list = []
        team_list = [[] for i in range(self.node_num-2)]
        y_sol = np.zeros(self.node_num-2, dtype=np.float64)
        for i_place in range(place_num):
            route_node_list.append(self.start_node)
            node_id = place_perm[i_place] 
            route_node_list.append(node_id)
            y_sol[node_id] = 1.0
            route_node_list.append(self.start_node)
        route_time_list = []

        assert len(route_node_list) > 2, 'OrtoolRoutingSolver.get_random_plan: An empty route!'
        route_time = 0.0
        route_time_list.append([route_time])
        for i_node in range(len(route_node_list) - 1):
            node_i = route_node_list[i_node]
            node_j = route_node_list[i_node+1]
            route_time += edge_time[node_i,node_j] + node_time[node_i]
            route_time_list.append(route_time)
            return route_node_list, route_time_list, y_sol

    def get_plan(self, flag_sub_solver = False, flag_verbose = False):
        # Output the plans
        """Prints solution on console."""
        route_node_list = []
        route_time_list = []
        team_list = [[] for i in range(self.node_num-2)]

        if not flag_sub_solver:
            solution = self.solution
            solver = self.solver
            manager = self.manager
            time_dimension = solver.GetDimensionOrDie('Time')
            if flag_verbose:
                print(f'Objective: {solution.ObjectiveValue()}')
        total_max_time = 0
        
        route_node = []
        route_time = []
        if flag_sub_solver:
            solution = self.sub_solution
            solver = self.sub_solver
            manager = self.sub_manager
            index = solver.Start(0)
            time_dimension = solver.GetDimensionOrDie('Time')
        else:
            index = solver.Start(0)
        while not solver.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            node_id = manager.IndexToNode(index)
            temp_min_time = solution.Min(time_var)
            temp_max_time = solution.Max(time_var)
            index = solution.Value(solver.NextVar(index))
            route_node.append(node_id)
            route_time.append(temp_min_time)
        time_var = time_dimension.CumulVar(index)
        node_id = manager.IndexToNode(index)
        temp_min_time = solution.Min(time_var)
        temp_max_time = solution.Max(time_var)
        if temp_min_time > total_max_time:
            total_max_time = temp_min_time
        route_node.append(node_id)
        route_time.append(temp_min_time)
        route_node_list.append(route_node)
        route_time_list.append(route_time)

        if flag_verbose:
            print('time_var = ', temp_min_time)
        if flag_verbose:
            print('Max time of all routes: {}min'.format(total_max_time))

        y_sol = np.zeros(self.node_num-2, dtype=np.float64)
        for i in range(self.node_num-2):
            y_sol[i] = 1.0
        # print(route_node_list)
        # print(route_time_list)
        # print(team_list)
        return route_node_list, route_time_list, y_sol
