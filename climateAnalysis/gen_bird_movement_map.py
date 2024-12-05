import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import xarray as xr
import pandas as pd
from matplotlib.patches import Rectangle
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Circle


def bird_mov_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse, bird_data_file, bounds):
    '''
    Generate bird movement map overlaid over temp maps

    Input:
    inp_path: Path containing TAVG .csv files in the format - startyr-month-date_to_endyr-month-date.csv
    start_year: Starting year
    end_year: Ending year
    duration_gap: Intervals of time for heatmap gen
    months_to_analyse: Array of months index starting from 1
    bird_data_file: bird location data per year slot
    bounds: map bounds list containing lat min, lat max, lon min, lon max

    Output:
    Dumps output heatmap images in the inp path under results folder

    '''
    assert(isinstance(inp_path, str) and isinstance(start_year, int) and isinstance(end_year, int))
    assert(isinstance(duration_gap, int) and isinstance(months_to_analyse, list) and isinstance(bird_data_file, str))
    assert(isinstance(bounds, list) and len(bounds) == 4)

    #Regions in the North America
    regions = {
        "Northeast (Cold/Continental)": {"lat": (36, 45), "lon": (-80, -71)},
        "Southeast (Humid Subtropical)": {"lat": (25, 36), "lon": (-90, -80)},
        "Midwest (Continental)": {"lat": (36, 49), "lon": (-95, -80)},
        "Great Plains (Semi-arid)": {"lat": (30, 49), "lon": (-105, -95)},
        "Southwest (Arid/Desert)": {"lat": (31, 37), "lon": (-115, -105)},
        "Rocky Mountain (Highland)": {"lat": (32, 49), "lon": (-115, -105)},
        "Pacific Northwest (Oceanic)": {"lat": (42, 49), "lon": (-123, -117)},
        "West Coast (Mediterranean)": {"lat": (32, 42), "lon": (-123, -116)}
    }

    months_str = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August','Sept', 'Oct','Nov','Dec']
    cols = ['time', 'lat', 'lon', 'tavg']

    #Plot Temperature Characteristics
    midpoint = 0
    vmin =-3
    vmax = 12
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=midpoint)


    num_slots = end_year - start_year // duration_gap
    read_chunksize = 5000000
    target_df = pd.DataFrame()
    ref = pd.DataFrame()

    df_goose = pd.read_pickle(bird_data_file)

    #Collect for ref and store it
    ref = pd.DataFrame()
    #Building Ref
    for chunk in pd.read_csv(inp_path + r'\\1960-12-31_to_1965-12-31.csv',usecols=cols, read_chunksize=read_chunksize):
        chunk['DATE'] = pd.to_datetime(chunk['time'], format='mixed')  # Convert, setting invalid formats to NaT
        chunk['YEAR'] = chunk['DATE'].dt.year
        chunk['MONTH'] = chunk['DATE'].dt.month
        chunk = chunk.reset_index()
        if (not chunk.empty):
            if (chunk['YEAR'][0] < 1961 and chunk['YEAR'][-1] < 1961):
                continue
            elif (chunk['YEAR'][0] > 1961):
                continue
            #print(chunk.head())
            ref = pd.concat([ref, chunk[chunk['YEAR'] == 1961]], axis = 0)

    #For each duration slot
    for slot_num in range(0, num_slots):
        target_year = [val for val in range(start_year+duration_gap*slot_num+1, start_year+duration_gap*slot_num+duration_gap+1)]
        inp_read_file_name = str(target_year[0]-1) + '-12-31_to_' + str(target_year[-1]) + '-12-31.csv'

        for chunk in pd.read_csv(inp_path + r'\\' + inp_read_file_name,usecols=cols, read_chunksize=read_chunksize):
            chunk['DATE'] = pd.to_datetime(chunk['time'], format='mixed')  # Convert, setting invalid formats to NaT
            chunk['YEAR'] = chunk['DATE'].dt.year
            chunk['MONTH'] = chunk['DATE'].dt.month
            chunk = chunk.reset_index()

            chunk = chunk[chunk['YEAR'].isin(target_year)]
            target_df = pd.concat([target_df,chunk], axis = 0)

        #For each month
        for month_index, month in enumerate(months_to_analyse):
            fig1, axes1 = plt.subplots(1, 1, figsize=(10, 6),subplot_kw={'projection': ccrs.PlateCarree()}) # Only one axis needed
            folder_name = inp_path + r'\\' + r"Results_2_Marked\\" + months_str[month-1]
            os.makedirs(folder_name, exist_ok=True)
            target_month = month
            target_month_df = target_df[target_df['MONTH'] == target_month]
            target_df = target_df[(target_df['lat'] >= bounds[0]) & (target_df['lat'] <= bounds[1]) &
                    (target_df['lon'] >= bounds[2]) & (target_df['lon'] <= bounds[3])]

            target_month_df = target_month_df.groupby(['lat','lon'])['tavg'].agg(['max', 'min']).reset_index()
            target_month_df.rename(columns={'max': 'tavg_max', 'min': 'tavg_min'}, inplace=True)
            target_month_df.head()
            target_month_df.head()
            ref_month = ref[ref['MONTH'] == month]

            #Merge ref and target data
            merged_df = pd.merge(ref_month, target_month_df, on=['lat', 'lon'], how='outer', suffixes=('_df1', '_df2'))
            merged_df.head()
            #print(local_merged_df.head())

            merged_df['tavg_diff_max'] = merged_df['tavg_max'] - merged_df['tavg']
            merged_df['tavg_diff_min'] = merged_df['tavg_min'] - merged_df['tavg']
            merged_df['tavg_diff_max'] = merged_df['tavg_diff_max'].fillna(0)
            merged_df['tavg_diff_min'] = merged_df['tavg_diff_min'].fillna(0)
            merged_df.head()

            ax = axes1
            ax.set_extent([-94, -78, 35, 50], crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.STATES, edgecolor="gray",linewidth=0.2)

            # Scatter plot of 'tavg_diff'
            sc = ax.scatter(
                merged_df['lon'],
                merged_df['lat'],
                c=merged_df['tavg_diff_max'],
                cmap='coolwarm',
                s=50,
                alpha=0.6,
                norm=norm,
                transform=ccrs.PlateCarree(),
            )

            for index, row in df_goose.iterrows():
                if (int(row['EVENT_YEAR']) == target_year[-1] and int(row['EVENT_MONTH']) == month):
                    circle = Circle(
                        xy=(row['LON_DD'], row['LAT_DD']),
                        radius=0.25,
                        edgecolor='black',
                        facecolor='green',
                        alpha=0.5,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.add_patch(circle)

            # Set title for the plot
            ax.set_title(f"{months_str[target_month-1]} - Temperature Difference", fontsize=10)

            # Colorbar
            cbar_ax1 = fig1.add_axes([0.92, 0.15, 0.02, 0.7])

            fig1.colorbar(sc, cax=cbar_ax1, label="Temperature Difference (°C)")

            fig1.tight_layout(rect=[0, 0, 0.9, 0.95])
            fig1.suptitle("Avg Temperature Change (Ref - 1960)", fontsize=16)
            output_img_name = folder_name + r'\\' + 'TAVG_Diff_' + str(slot_num * duration_gap+start_year+1) + '_' + str(slot_num * duration_gap + start_year + duration_gap) + '.png'
            fig1.savefig(output_img_name)
            print(output_img_name)