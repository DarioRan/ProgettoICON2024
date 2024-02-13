import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pulp
import math
from matplotlib import pyplot as plt


class DriverAssignmentProblem:
    def __init__(self, drivers_data, order_location):
        """
        Constructs all the necessary attributes for the driver assignment problem.

        Parameters:
            drivers_data: dataframe containing drivers data.
            order_location (tuple): The (latitude, longitude) of the order's location.
        """
        self.drivers_df = drivers_data
        self.order_location = order_location
        self.prob = None
        self.driver_vars = None

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Euclidean distance between two points in latitude and longitude.

        Parameters:
            lat1, lon1 (float): Latitude and longitude of the first point.
            lat2, lon2 (float): Latitude and longitude of the second point.

        Returns:
            float: Euclidean distance between two points.
        """
        return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    def create_problem(self):
        """
        Creates the linear programming problem for the driver assignment.

        The objective is to minimize the total distance of the assigned driver from the order's location.
        """
        # Crea il problema di minimizzazione con PuLP
        self.prob = pulp.LpProblem("DriverAssignmentProblem", pulp.LpMinimize)

        # Crea le variabili di decisione per l'assegnazione dei driver all'ordine
        self.driver_vars = [pulp.LpVariable(f'driver_{row.Driver_ID}', cat='Binary') for index, row in self.drivers_df.iterrows()]

        # Aggiungi la funzione obiettivo: minimizzare la distanza complessiva per l'ordine
        distances = [self.euclidean_distance(row.Latitude, row.Longitude, self.order_location[0], self.order_location[1]) for index, row in
                     self.drivers_df.iterrows()]
        self.prob += pulp.lpSum([distances[i] * self.driver_vars[i] for i in range(len(self.driver_vars))]), "Total_Distance"

        # Vincolo: solo un driver può essere assegnato all'ordine
        self.prob += pulp.lpSum(self.driver_vars) == 1, "OneDriver"

        # Vincolo di disponibilità: solo i driver disponibili possono essere assegnati
        for i, row in self.drivers_df.iterrows():
            if row.Availability == 'Unavailable':
                self.prob += self.driver_vars[i] == 0, f"Unavailable_{row.Driver_ID}"

    def solve_problem(self):
        """
       Solves the driver assignment problem using the branch and bound algorithm.
       """
        no_print = pulp.LpSolverDefault.msg
        pulp.LpSolverDefault.msg = 0

        self.prob.solve()

        pulp.LpSolverDefault.msg = no_print

    def get_assigned_driver_details(self):
        """
        Retrieves the details of the assigned driver.

        Returns:
            dict: A dictionary with the status of the problem, driver ID, driver's location, and distance to the order.
            If no driver is assigned, returns the status and an error message.
        """
        # Stampiamo i risultati
        status = pulp.LpStatus[self.prob.status]
        assigned_driver_id = None
        assigned_driver_distance = None

        # Troviamo il driver assegnato e la distanza minima
        for v in self.prob.variables():
            if v.varValue > 0:
                assigned_driver_id = int(v.name.split('_')[-1])
                assigned_driver_distance = pulp.value(self.prob.objective)
                break

        if assigned_driver_id is not None:
            driver_row = self.drivers_df[self.drivers_df['Driver_ID'] == assigned_driver_id].iloc[0]
            return {
                'status': status,
                'driver_id': assigned_driver_id,
                'driver_latitude': driver_row['Latitude'],
                'driver_longitude': driver_row['Longitude'],
                'distance': assigned_driver_distance
            }
        else:
            return {'status': "no_assignments", 'error': 'No driver could be assigned.'}

    def plot_assignments(self):
        """
        Plots the drivers' locations and the order location on a map.
        The assigned driver's location is highlighted.
        """
        gdf = gpd.GeoDataFrame(self.drivers_df,
                               geometry=gpd.points_from_xy(self.drivers_df.Longitude, self.drivers_df.Latitude))

        order_gdf = gpd.GeoDataFrame(
            [{'Order_ID': 'Order', 'geometry': Point(self.order_location[1], self.order_location[0])}])

        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(ax=ax, color='blue', markersize=5, label='Driver')
        order_gdf.plot(ax=ax, color='red', markersize=100, label='Order')

        assigned_details = self.get_assigned_driver_details()
        if assigned_details['status'] == 'Optimal' and 'driver_id' in assigned_details:
            assigned_driver = gdf[gdf['Driver_ID'] == assigned_details['driver_id']]
            assigned_driver.plot(ax=ax, color='green', markersize=100, label='Assigned Driver')

        plt.legend()
        plt.savefig('driver_assignments.png')