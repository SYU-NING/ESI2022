from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def retreive_depot_customers():
    ''' create a dictionary recording for each open warehouse, the group of allocated customers;
    '''
    customer_groups_dict = {}
    for k in K:                                 # for all warehouses: 
        if y[k].X > 0.99:                       # if a potential warehouse is opened: 
            print('* Open Warehouse: ', k)
            assigned_customers = []
            for j in J:                         # for all customers: find those allocated to warehouse k
                if x[(k,j)].X > 0.99:
                    assigned_customers.append(j)
            print('* Allocated customers: ', assigned_customers)
            customer_groups_dict[k] = assigned_customers
    return customer_groups_dict


def extract_distance_matrix(warehouse, allocated_customers): 
    ''' create distance matrix for OR tools software; 
        warehouse: index from 0, total 430 potential location; 
        customers: index from 0, total 430;
    '''
    # matrix size = warehouse + customers
    matrix_size = len(allocated_customers)+1

    # form initial travel distance save in list format: 
    distance_matrix = np.zeros((matrix_size, matrix_size)).tolist()
    
    # enumerate through the pairs of node (start, end), store in distance matrix; Row = start node, Col = end node 
    allocated_customers.insert(0, warehouse)
    for i, cust in enumerate(allocated_customers[:-1]):
        for j in range(i+1, len(allocated_customers)):
            next_cust = allocated_customers[j]
            distance_matrix[i][j] = distance_dd.iloc[cust, next_cust+1]
            distance_matrix[j][i] = distance_dd.iloc[next_cust, cust+1]
    return distance_matrix  


def create_data_model(customer_groups_dict, warehouse, num_vehicles):
    ''' warehouse index = 4, actual warehouse number = 4+1 = 5 
    '''
    # extract the opened warehouse's allocated customers: 
    allocated_customers = customer_groups_dict[warehouse]
    
    # compute total demand associated with each postcode:
    g1 = postcode['Group 1'].tolist()
    g2 = postcode['Group 2'].tolist()
    g3 = postcode['Group 3'].tolist()
    g4 = postcode['Group 4'].tolist()
    total_demand =[g1[j]+g2[j]+g3[j]+g4[j] for j in range(len(postcode))]

    # create data format for OR Tools: 
    data = {}
    data['distance_matrix'] = extract_distance_matrix(warehouse, allocated_customers)
    data['demands'] = [0] + [total_demand[j] for j in allocated_customers]
    data['depot'] = 0
    data['num_vehicles'] = num_vehicles
    data['vehicle_capacities'] = 3500 * num_vehicles 
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))

def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

def demand_callback(from_index):
    '''Returns the demand of the node.'''
    # Convert from routing variable Index to demands NodeIndex.
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]


# 1. get dictionary with {open warehouse: group of allocated customers}
#customer_groups_dict = retreive_depot_customers()
customer_groups_dict = {4:[1,2,438]}

# 2. extract distance matrix from data, create ortools data format: 
for warehouse in customer_groups_dict.keys(): 
    
    # self-defined number of vehicles:
    num_vehicles = 2

    # instantiate the data problem.
    data = create_data_model(customer_groups_dict, warehouse, num_vehicles)

    # create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # create and register a transit callback.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # add capacity constraint.
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)

