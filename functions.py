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



class HaggisWhiskey:

    def __init__(self):
        ## read in as dataframe from excel: 
        self.distance_dd = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance District-District.xlsx', sheet_name='Sheet1')
        self.distance_sd = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance Supplier-District.xlsx', sheet_name='Sheet1')
        self.distance_ss = pd.read_excel('/Users/shunee/github/ESI2022/data/Distance Supplier-Supplier.xlsx', sheet_name='Sheet1')
        self.postcode = pd.read_excel('/Users/shunee/github/ESI2022/data/Postcode Districts.xlsx', sheet_name='Postcode Districts - Class')
        self.potential_location = pd.read_excel('/Users/shunee/github/ESI2022/data/Potential Locations.xlsx', sheet_name='Postcode Districts - Class')
        self.supplier = pd.read_excel('/Users/shunee/github/ESI2022/data/Suppliers.xlsx', sheet_name='SuppliersClass')

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
    
