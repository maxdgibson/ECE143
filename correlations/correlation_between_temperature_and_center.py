import pandas as pd
from scipy.stats import mode
from geopy.geocoders import Nominatim
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from collections import OrderedDict
from scipy.optimize import minimize
import sklearn
import math
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

columns_to_load = [ "EVENT_DAY", "EVENT_YEAR", "EVENT_MONTH",
                   'ISO_COUNTRY', 'ISO_SUBDIVISION','LAT_DD','LON_DD']
eastern_states = [
        'ME', 'VT', 'NH', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA',
        'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'AL', 'TN', 'KY'
]

def load_and_clean_data(birdsFilePath,temperatureFilePath):
    df = pd.read_csv(birdsFilePath, usecols=columns_to_load)
    filtered_df = df[(df['ISO_COUNTRY'] == 'US')
                     & (1 <= df['EVENT_MONTH'])
                     & (df['EVENT_MONTH'] <= 12)
                     ].dropna()
    grouped_by_states = filtered_df.groupby('ISO_SUBDIVISION')
    count_by_state = {}
    for state, group in grouped_by_states:
        count_by_state[state] = len(group)
    sorted_by_count = OrderedDict(sorted(count_by_state.items(), key=lambda x: x[1], reverse=True))
    grouped_by_month = filtered_df[(1962 <= filtered_df['EVENT_YEAR'])
                                   & (filtered_df['EVENT_YEAR'] <= 2018)
                                   & (filtered_df['ISO_SUBDIVISION'].isin(eastern_states))].groupby(
        ["EVENT_YEAR", "EVENT_MONTH"])
    temperatures = np.genfromtxt(temperatureFilePath, delimiter=',', skip_header=1, usecols=[1])

    return filtered_df, sorted_by_count, grouped_by_month, temperatures

def counts_by_States(sorted_by_count):
    fig, ax = plt.subplots(figsize=(15, 10))
    m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66, resolution='i', ax=ax)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue', zorder=0)
    m.drawcountries()
    m.drawstates()

    shp_info = m.readshapefile('../datasets/geodata/s_05mr24', name='states', drawbounds=True)

    colors = ['yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list(None, colors)

    norm = mcolors.Normalize(vmin=min(sorted_by_count.values()), vmax=max(sorted_by_count.values()))

    for info, shape in zip(m.states_info, m.states):
        state_abbr = info['STATE']
        if state_abbr in sorted_by_count:
            count = sorted_by_count[state_abbr]
            color = cmap(norm(count))
        else:
            color = 'white'
        poly = plt.Polygon(shape, facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(poly)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, aspect=20, label='Count')

    plt.title('Counts by States')
    plt.show()


def weiszfeld_algorithm(grouped_by_month, max_iter=100, tol=1e-6, min_dist=1e-5):
    def update_center(points, current_center):
        distances = np.linalg.norm(points - current_center, axis=1)

        weights = 1 / (distances + min_dist)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)

        new_center = np.sum(weights[:, None] * points, axis=0) / np.sum(weights)
        return new_center

    centers = []
    for group_keys, group_data in grouped_by_month:
        points = group_data[['LON_DD', 'LAT_DD']].to_numpy()

        if len(points) == 0:
            print(f"Warning: No data for group {group_keys}")
            centers.append(None)
            continue

        current_center = np.mean(points, axis=0)

        for iteration in range(max_iter):
            new_center = update_center(points, current_center)
            diff = np.linalg.norm(new_center - current_center)

            if diff < tol:
                break

            current_center = new_center

        centers.append(current_center)
    centers_lon = [i[0] for i in centers]
    centers_lat = [i[1] for i in centers]

    return centers_lon, centers_lat

def scatter_graphs(temperatures, centers_lat, centers_lon):
    combinations = list(zip(temperatures, centers_lat, centers_lon))
    sorted_combinations = sorted(combinations, key=lambda x: x[0])
    sorted_temperatures = [i[0] for i in sorted_combinations]
    sorted_lat = [i[1] for i in sorted_combinations]
    sorted_lon = [i[2] for i in sorted_combinations]

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_temperatures, sorted_lat, label='Latitude', color='yellow', marker='p')
    plt.xlabel('Temperature')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_temperatures, sorted_lon, label='Longitude', color='green', marker='*')
    plt.xlabel('Temperature')
    plt.ylabel('Longitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    return sorted_temperatures, sorted_lat, sorted_lon

def Correlation(sorted_temperatures, sorted_lat, sorted_lon):
    mi_lon = mutual_info_regression(np.array(sorted_temperatures)[:, None], np.array(sorted_lon))
    mi_lat = mutual_info_regression(np.array(sorted_temperatures)[:, None], np.array(sorted_lat))

    corrcoef_lat = np.corrcoef(sorted_temperatures, sorted_lat)[0, 1]
    corrcoef_lon = np.corrcoef(sorted_temperatures, sorted_lon)[0, 1]

    return corrcoef_lat, corrcoef_lon, mi_lat, mi_lon

def MSE(preds,labels):
    numer = 0
    for pred, label in zip(preds, labels):
        numer += (pred-label)**2
    return numer/len(labels)

def linear(sorted_temperatures, sorted_lat, sorted_lon):
    features = list(zip([1] * len(sorted_temperatures), sorted_temperatures))
    model = sklearn.linear_model.LinearRegression()
    model.fit(features, sorted_lat)
    preds = model.predict(features)
    X = np.linspace(20, 80, 100).reshape(-1, 1)
    feature = np.hstack([np.ones((X.shape[0], 1)), X])
    y = model.predict(feature)
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, color='red', linewidth=4)
    plt.scatter(sorted_temperatures, sorted_lat, label='Latitude', color='yellow', marker='p')
    plt.xlabel('Temperature')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    mse1 = MSE(preds, sorted_lat)

    features = list(zip([1] * len(sorted_temperatures), sorted_temperatures))
    model = sklearn.linear_model.LinearRegression()
    model.fit(features, sorted_lon)
    preds = model.predict(features)
    X = np.linspace(20, 80, 100).reshape(-1, 1)
    feature = np.hstack([np.ones((X.shape[0], 1)), X])
    y = model.predict(feature)
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, color='red', linewidth=4)
    plt.scatter(sorted_temperatures, sorted_lon, label='Longitude', color='green', marker='*')
    plt.xlabel('Temperature')
    plt.ylabel('Longitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    mse2 = MSE(preds, sorted_lon)

    return mse1, mse2

def migration_route(temperatures, centers_lat, centers_lon):
    # %%
    plt.figure(figsize=(15, 15))
    m = Basemap(projection='merc',
                llcrnrlat=24, urcrnrlat=48,
                llcrnrlon=-85, urcrnrlon=-67,
                resolution='i')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawrivers()

    m.drawparallels(
        np.linspace(24, 48, 10),
        labels=[1, 0, 0, 0],
        fontsize=10, color='gray'
    )
    m.drawmeridians(
        np.linspace(-85, -67, 10),
        labels=[0, 0, 0, 1],
        fontsize=10, color='gray'
    )

    colors = ['red', 'green', 'yellow', 'magenta', 'orange', 'cyan']
    years = [1973, 1980, 1994, 2015]
    for j in range(len(years)):
        x = []
        y = []
        for i in range((years[j] - 1962) * 12, (years[j] + 1 - 1962) * 12):
            x_i, y_i = m(centers_lon[i], centers_lat[i])
            x.append(x_i)
            y.append(y_i)
            # m.plot(x, y, color='red', linewidth=2, marker='o', markersize=8)

        f = interp1d(np.arange(len(x)), np.array(list(zip(x, y))), kind='cubic', axis=0)

        new_points = f(np.linspace(0, len(x) - 1, 100))

        m.plot(new_points[:, 0], new_points[:, 1], color=colors[j], linewidth=2, label=years[j])
        m.scatter(new_points[0, 0], new_points[0, 1], color=colors[j], marker='o', s=100, label='Start')

        m.scatter(new_points[-1, 0], new_points[-1, 1], color=colors[j], marker='s', s=100, label='End')
    plt.title('center migration route in 1973,1980,1994,2015')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))

    for j in range(len(years)):
        plt.plot([i for i in range(1, 13)], temperatures[(years[j] - 1962) * 12:(years[j] + 1 - 1962) * 12],
                 label=years[j], color=colors[j], marker='o')

    plt.xlabel('Month')
    plt.ylabel('Temperatures')
    plt.title('Temperatures by month in 1973,1980,1994,2015')

    plt.legend()

    plt.grid(True)

    plt.show()





