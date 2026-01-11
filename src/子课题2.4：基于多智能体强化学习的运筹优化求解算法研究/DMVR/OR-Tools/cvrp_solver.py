"""Capacited Vehicles Routing Problem (CVRP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import argparse
parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--vec_num", type=int, default=7, help="random seed")
parser.add_argument("--customer_nodes", type=int, default=50, help="random seed")
args = parser.parse_args()

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.3, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def plot(coords, demand, locations, demands, traj_color):
    x,y = coords.T
    lc = colorline(x,y,z = np.array([traj_color]*len(x)), cmap='jet')
    plt.axis('square')
    x, y =locations.T
    h = demands/4
    h = np.vstack([h*0,h])
    plt.errorbar(x,y,h,fmt='None',elinewidth=2)

    return lc

def plt_cvrp(locations, demands, vec_num, resulting_traj_with_depot, tot_len=0):
    #! zjk add
    x = vec_num + 1
    traj_color = [j / ( x- 1) for j in range(x)]
    
    tc=0
    resulting_traj_with_depot_per_agent = [0]
    for i in range(len(resulting_traj_with_depot)):
        resulting_traj_with_depot_per_agent.append(resulting_traj_with_depot[i])
        if resulting_traj_with_depot[i] == 0:
            # plot(nodes_coordinates[resulting_traj_with_depot_per_agent], obs['demand'][resulting_traj_with_depot_per_agent[0],resulting_traj_with_depot_per_agent[-1]], traj_color[tc])
            plot(locations[resulting_traj_with_depot_per_agent], demands[resulting_traj_with_depot_per_agent[0]:resulting_traj_with_depot_per_agent[-1]],locations, demands, traj_color[tc])
            tc += 1
            resulting_traj_with_depot_per_agent = [0]
    
    plt.title('OR-Tools, Agent_num={}, Tot_len = {}'.format(vec_num, tot_len))
    plt.savefig("/root/git_zjk/RLOR/OR-Tools/solver_cvrp.png")
    #! zjk add


# 构造一个函数，输入是一个大小为m*2的列表，表示m个点的位置，输出是这m个点的距离矩阵，输出形式是列表
def calculate_distance_matrix(locations):
    m = len(locations)
    distance_matrix = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            x1, y1 = locations[i]
            x2, y2 = locations[j]
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            distance_matrix[i][j] = round(distance)
    return distance_matrix


def create_data_model(distance_matrix, demands, vec_num, vehicle_capacities):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["demands"] = demands
    data["num_vehicles"] = vec_num
    data["vehicle_capacities"] = [vehicle_capacities] * data["num_vehicles"]
    data["depot"] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    result_route = [[]]*data["num_vehicles"]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            result_route[vehicle_id].append(node_index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance/1e7}m\n"
        plan_output += f"Load of the route: {route_load/1e3}\n"
        # print(plan_output)
        total_distance += route_distance
        total_load += route_load
    # print(f"Total distance of all routes: {total_distance/1e7}m")
    # print(f"Total load of all routes: {total_load/1e3}")
    
    result_route[0].append(0)
    # print("result_route:", result_route[0][2:])

    # plt_cvrp(locations_arry, demands_arry, data["num_vehicles"], result_route[0][2:], -total_distance/1e7 )
    
    return total_distance
    


def main(distance_matrix, demands, vec_num, vehicle_capacities):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(distance_matrix, demands, vec_num, vehicle_capacities)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        total_distance = print_solution(data, manager, routing, solution)
    
    return total_distance


if __name__ == "__main__":
    import time
    import pickle
    time_start = time.time()
    if args.customer_nodes==50: 
        locations_10000 = pickle.load(open("./data/vrp/vrp50_validation_seed324750.pkl", "rb"))
        locations_10000_arry = np.array(locations_10000)
        vec_num=20
    
    
    for i in range(10000):
        locations = np.concatenate((np.array(locations_10000_arry[i][0])[np.newaxis,:], np.array(locations_10000_arry[i][1])),0)*1e7
        demands = locations_10000_arry[i][2]
        demands.insert(0, 0)
        vehicle_capacities = int(locations_10000_arry[i][3])
        
        
        distance_matrix = calculate_distance_matrix(locations.tolist())

        total_distance = main(distance_matrix, demands, vec_num, vehicle_capacities)/1e7
        total_distance_list.append(total_distance)

        total_distance_arry = np.array(total_distance_list)

        end_start = time.time()
    
    end_start = time.time()
    print("CVRP-{}地图的平均距离为:{}".format(args.customer_nodes, total_distance_arry.mean()))
    print("总共用时：", end_start-time_start)

    
    





