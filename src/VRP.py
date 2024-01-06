from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import os

#Enter the list of locations where VRP needs to be implemented
loc = [[249, 25, 255, 30, 32, 231], [0, 168, 207, 159, 167, 183, 76, 73, 214], [8, 55, 19, 161, 272, 163, 21, 23, 13, 16], [102, 123, 124, 131, 113, 143, 144, 108, 132, 110, 87, 112, 97, 243, 241, 238]]#user input
global b
b = []

def distMat(loc):
    loc1 = loc.copy()
    loc1 = np.insert(loc,0,328) # depot location
    dist_mat = "input/Distances.xlsx"
    dist_mat = os.path.abspath(dist_mat)
    df = pd.read_excel(dist_mat, sheet_name=2, header=None)
    df = df.loc[loc1,loc1] # to extract distance matrix for specific locations alone
    m = df.to_numpy() # to create a numpy array for distance matrix
    return m

def create_data_model(dmat):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = dmat
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} metres'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = '\n'
    route_distance = 0
    global a
    a = []
    while not routing.IsEnd(index):
        #plan_output += ' {} , '.format(manager.IndexToNode(index))
        a.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    a.pop(0)
    #plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Route distance: {}metres'.format(route_distance)
    print(plan_output)
    return a

def vrp(loc):
    pl_list = []
    for i in range(len(loc)):
        pl = loc[i]
        pl_list.append(distMat(pl))
    
    for i in range(len(loc)):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = create_data_model(pl_list[i])

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(manager, routing, solution)

        values = loc[i]
        order = a
        Order_visit = sort_list(values, order)
        Order_visit.append(328)
        Order_visit.insert(0,328)
        Visit_Order(Order_visit)
        print('Tour' + ' ' + str(i+1)+ ':' , Order_visit) 
    print (b)

def Visit_Order(Order_visit):
    b.append(Order_visit)
    
def sort_list(values, order):
    # Zip the values and order lists together
    zipped_lists = zip(values, order)

    # Sort the zipped list by the order of the elements in the second list
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[1])

    # Unzip the sorted list and return the first list (the values list)
    sorted_values, _ = zip(*sorted_zipped_lists)

    sorted_values = np.array(sorted_values)
    sorted_values = sorted_values.tolist()
    return sorted_values

vrp(loc)