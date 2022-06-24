import random
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame

import matplotlib.pyplot as plt
from shapely.geometry import Point
import convertbng     # https://pypi.org/project/convertbng/
from convertbng.util import convert_bng, convert_lonlat
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class HaggisWhiskey:

    def __init__(self):
        ## read in as dataframe from excel: 
        self.distance_dd = pd.read_excel('/data/Distance District-District.xlsx', sheet_name='Sheet1')
        self.distance_sd = pd.read_excel('/data/Distance Supplier-District.xlsx', sheet_name='Sheet1')
        self.distance_ss = pd.read_excel('/data/Distance Supplier-Supplier.xlsx', sheet_name='Sheet1')
        self.postcode = pd.read_excel('/data/Postcode Districts.xlsx', sheet_name='Postcode Districts - Class')
        self.potential_location = pd.read_excel('/data/Potential Locations.xlsx', sheet_name='Postcode Districts - Class')
        self.supplier = pd.read_excel('/data/Suppliers.xlsx', sheet_name='SuppliersClass')

        ## define production level and categorical demands:
        self.production = self.supplier['Production volume']
        self.group_1 = self.postcode['Group 1'].tolist()
        self.group_2 = self.postcode['Group 2'].tolist()
        self.group_3 = self.postcode['Group 3'].tolist()
        self.group_4 = self.postcode['Group 4'].tolist()
        print('Number of supplier = {}, number of customer postcode = {}'.format(len(self.production), len(self.group_1)))

        ## convert (easting, northing) coordinate to (latitute, longitude) for warehouses, producers, customers; 
        lat_supplier, lon_supplier = self.convert_to_latlon(self.supplier)
        lat_potential, lon_potential = self.convert_to_latlon(self.potential_location)
        lat_demand, lon_demand = self.convert_to_latlon(self.postcode)
        self.lat_supplier = lat_supplier
        self.lon_supplier = lon_supplier
        self.lat_potential = lat_potential
        self.lon_potential = lon_potential
        self.lat_demand = lat_demand
        self.lon_demand = lon_demand


    def convert_to_latlon(self, dataframe): 
        eastings, northings = dataframe['X (Easting)'].tolist(), dataframe['Y (Northing)'].tolist()
        lon, lat = convert_lonlat(eastings, northings)
        return lat, lon
        

    def plot_geographical_nodes(self):
        insert_central_location = [56.996140, -3.902614] # EDI:(Lat=55.9533, Lon=-3.1883)

        Edinburgh_map = folium.Map(location = insert_central_location, 
                                control_scale = True,
                                zoom_start = 7, 
                                tiles = 'Stamen Terrain')

        folium.Marker([55.9533, -3.1883], popup='Edinburgh').add_to(Edinburgh_map)

        # When clicking the map, we could obtain the longitude and latitude
        Edinburgh_map.add_child(folium.LatLngPopup())


        fg1 = folium.FeatureGroup('Supplier Location')
        fg2 = folium.FeatureGroup('Potential Warehouse Location')
        fg3 = folium.FeatureGroup('Customer Demand Location')

        ## supplier: 
        for i in range(len(self.lat_supplier)):
            fg1.add_child(
                folium.CircleMarker(radius = 1,
                                    location = [self.lat_supplier[i], self.lon_supplier[i]],
                                    color = 'red',
                                    fill = True,
                                    fill_color = 'red')
            )

        ## potential location: 
        for i in range(len(self.lat_potential)):
            fg2.add_child(
                folium.CircleMarker(radius = 1,
                                    location = [self.lat_potential[i], self.lon_potential[i]],
                                    color = 'blue',
                                    fill = True,
                                    fill_color = 'blue')
            )

        ## customer postcode location: 
        for i in range(len(self.lat_demand)):
            fg3.add_child(
                folium.CircleMarker(radius = 1,
                                    location = [self.lat_demand[i], self.lon_demand[i]],
                                    color = 'green',
                                    fill = True,
                                    fill_color = 'green')
            )

        Edinburgh_map.add_child(fg1)  
        Edinburgh_map.add_child(fg2)  
        Edinburgh_map.add_child(fg3)  

        folium.LayerControl().add_to(Edinburgh_map)
        
        return Edinburgh_map


    def plot_heat_map_bw(self, latitude, longtitude, dataframe): 

        max_value = max(dataframe['Group 1'].tolist())
        geo_stat_list_final = pd.DataFrame({'Lat':latitude,'Long':longtitude,'DemandTotal': dataframe['Group 4'].tolist()})
        geo_stat_list_final['weight'] = geo_stat_list_final['DemandTotal'] / max_value               #scale to [0,1]

        max_value = 1

        fig = go.Figure(go.Densitymapbox(lat=geo_stat_list_final.Lat, 
                                        lon=geo_stat_list_final.Long,
                                        z=geo_stat_list_final.weight,
                                        radius=10,
                                        colorscale=[[0.0, 'blue',],[0.3*max_value,'lime'],[0.5*max_value,'yellow'],[0.7*max_value,'orange'],[1.0*max_value, 'red']],# custome colorscale
                                        zmin=0.0,
                                        zmax=1.0*max_value
                                        ))

        fig.update_layout(mapbox_style="carto-positron",
                        mapbox_center_lon= -3.902614,
                        mapbox_center_lat= 56.996140, 
                        mapbox_zoom=6)

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()


    def plot_heat_map_colour(self, latitude, longtitude, dataframe, group):
            
        #group = 'Production volume' #'Group 4'
        
        max_value = max(dataframe[group].tolist())
        geo_stat_list_final = pd.DataFrame({'Lat':latitude,'Long':longtitude,'DemandTotal': dataframe[group].tolist()})
        geo_stat_list_final['weight'] = geo_stat_list_final['DemandTotal'] / max_value               #scale to [0,1]

        def generateBaseMap(loc, zoom=7, tiles='OpenStreetMap', crs='ESPG2263'):
                return folium.Map(location=loc,
                                control_scale=True, 
                                zoom_start=zoom,
                                tiles=tiles)

        MAP_CENTRE = [56.996140, -3.902614]  
        base_map = generateBaseMap(MAP_CENTRE)

        map_values1 = geo_stat_list_final[['Lat','Long','weight']]
        data = map_values1.values.tolist()
                
        heat_map = HeatMap(data,gradient={0.1: 'orange', 0.3: 'orange', 0.5: 'red', 0.7: 'red', 1: 'purple'}, 
                        min_opacity=0.05, 
                        max_opacity=0.9, 
                        radius=15,
                        use_local_extrema=False).add_to(base_map)

        base_map.add_child(heat_map)
        return base_map


    def write_to_excel(self): 
        pass 
    
    
    


class ORTools(HaggisWhiskey): 

    def __init__(self, num_vehicles, vehicle_capacity): # num_vehicles=2, vehicle_capacity=1
        
        self.num_vehicles = num_vehicles            # self-defined number of vehicles
        self.vehicle_capacity = vehicle_capacity    # vehicle capacity (tons)

        self.distance_dd = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance District-District.xlsx', sheet_name='Sheet1')
        self.distance_sd = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance Supplier-District.xlsx', sheet_name='Sheet1')
        self.distance_ss = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance Supplier-Supplier.xlsx', sheet_name='Sheet1')
        self.postcode = pd.read_excel('/Users/shunee/github/ESI2022/data/Postcode Districts.xlsx', sheet_name='Postcode Districts - Class')
        self.potential_location = pd.read_excel('/Users/shunee/github/ESI2022/data/Potential Locations.xlsx', sheet_name='Postcode Districts - Class')
        self.supplier = pd.read_excel('/Users/shunee/github/ESI2022/data/Suppliers.xlsx', sheet_name='SuppliersClass')

        self.K = range(len(self.potential_location))
        self.J = range(len(self.postcode))


        # 1. get dictionary with {open warehouse: group of allocated customers}
        #self.customer_groups_dict = retreive_depot_customers()
        self.customer_groups_dict = {0:  [x for x in range(1,10)],
                                     #17: [x for x in range(17,20)],
                                     #20: [x for x in range(21,30)]
                                     }

        # 2. extract distance matrix from data, create ortools data format: 
        for warehouse in self.customer_groups_dict.keys(): 
            
            # instantiate the data problem.
            self.data = self.create_data_model(self.customer_groups_dict, warehouse, num_vehicles)

            # create the routing index manager.
            self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'], self.data['depot'])

            # create Routing Model.
            routing = pywrapcp.RoutingModel(self.manager)

            # create and register a transit callback.
            transit_callback_index = routing.RegisterTransitCallback(self.distance_callback)

            # define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # add capacity constraint.
            demand_callback_index = routing.RegisterUnaryTransitCallback(self.demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # null capacity slack
                    self.data['vehicle_capacities'],  # vehicle maximum capacities
                    True,  # start cumul to zero
                    'Capacity')

            # setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.FromSeconds(1)

            # solve the problem.
            solution = routing.SolveWithParameters(search_parameters)

            # print solution on console.
            if solution:
                print('..............................')
                print('Warehouse No. ', warehouse)
                print('Number of Vehicles: ', num_vehicles)
                print('..............................')
                print()
                self.print_solution(self.data, self.manager, routing, solution)


    def retreive_depot_customers(self, x, y):
        ''' create a dictionary recording for each open warehouse, the group of allocated customers;
        '''
        customer_groups_dict = {}
        for k in self.K:                                 # for all warehouses: 
            if y[k].X > 0.99:                            # if a potential warehouse is opened: 
                print('* Open Warehouse: ', k)
                assigned_customers = []
                for j in self.J:                         # for all customers: find those allocated to warehouse k
                    if x[(k,j)].X > 0.99:
                        assigned_customers.append(j)
                print('* Allocated customers: ', assigned_customers)
                customer_groups_dict[k] = assigned_customers
        return customer_groups_dict

    def extract_distance_matrix(self, warehouse, allocated_customers): 
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
                distance_matrix[i][j] = self.distance_dd.iloc[cust, next_cust+1]
                distance_matrix[j][i] = self.distance_dd.iloc[next_cust, cust+1]
        return distance_matrix  

    def create_data_model(self, customer_groups_dict, warehouse, num_vehicles):
        ''' warehouse index = 4, actual warehouse number = 4+1 = 5 
        '''
        # extract the opened warehouse's allocated customers: 
        allocated_customers = customer_groups_dict[warehouse]
        
        # compute total demand associated with each postcode:
        g1 = self.postcode['Group 1'].tolist()
        g2 = self.postcode['Group 2'].tolist()
        g3 = self.postcode['Group 3'].tolist()
        g4 = self.postcode['Group 4'].tolist()
        total_demand =[g1[j]+g2[j]+g3[j]+g4[j] for j in range(len(self.postcode))]

        # create data format for OR Tools: 
        data = {}
        data['distance_matrix'] = self.extract_distance_matrix(warehouse, allocated_customers)
        data['demands'] = [0] + [total_demand[j] for j in allocated_customers]
        data['depot'] = 0
        data['num_vehicles'] = num_vehicles
        #vehicle_capacity = 1
        data['vehicle_capacities'] = [self.vehicle_capacity * 1000_000] * num_vehicles 
        return data

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        #print(f'Objective: {solution.ObjectiveValue()}')
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = '*** Route for vehicle {}:\n'.format(vehicle_id + 1)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load/1000_000)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += ' {0} Load({1}) tons\n'.format(manager.IndexToNode(index), route_load/1000_000)  #
            plan_output += 'Distance of the route: {} km\n'.format(route_distance)
            plan_output += 'Load of the route: {} tons\n'.format(round(route_load/1000_000, 2))  #
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print('Total distance of all routes: {} km'.format(total_distance))
        print('Total load of all routes: {} tons'.format(round(total_load/1000_000, 2)))
        print()

    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]

    def demand_callback(self, from_index):
        '''Returns the demand of the node.'''
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.data['demands'][from_node]        
