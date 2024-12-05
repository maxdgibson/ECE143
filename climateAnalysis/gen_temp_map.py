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



def temp_heat_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse):
    '''
    Generate temperature heat maps

    Input:
    inp_path: Path containing TAVG .csv files in the format - startyr-month-date_to_endyr-month-date.csv
    start_year: Starting year
    end_year: Ending year
    duration_gap: Intervals of time for heatmap gen
    months_to_analyse: Array of months index starting from 1

    Output:
    Dumps output heatmap images in the inp path under results folder
    '''
    assert(isinstance(inp_path, str) and isinstance(start_year, int) and isinstance(end_year, int))
    assert(isinstance(duration_gap, int) and isinstance(months_to_analyse, list))

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

    #Building Ref data based on 1961
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

    #Building Target Data to compare with Ref in splits of duration gaps
    for slot_num in range(1, num_slots):
        target_year = [val for val in range(start_year+duration_gap*slot_num+1, start_year+duration_gap*slot_num+duration_gap+1)]
        inp_read_file_name = str(target_year[0]-1) + '-12-31_to_' + str(target_year[-1]) + '-12-31.csv'

        for chunk in pd.read_csv(inp_path + r'\\' + inp_read_file_name,usecols=cols, read_chunksize=read_chunksize):
            chunk['DATE'] = pd.to_datetime(chunk['time'], format='mixed')  # Convert, setting invalid formats to NaT
            chunk['YEAR'] = chunk['DATE'].dt.year
            chunk['MONTH'] = chunk['DATE'].dt.month
            chunk = chunk.reset_index()

            chunk = chunk[chunk['YEAR'].isin(target_year)]
            target_df = pd.concat([target_df,chunk], axis = 0)

        #Concentrating on particular month
        for month_index, month in enumerate(months_to_analyse):
            folder_name = inp_path + r"\Results\\" + months_str[month-1]
            os.makedirs(folder_name, exist_ok=True)
            target_month = month
            target_month_df = target_df[target_df['MONTH'] == target_month]
            target_month_df = target_month_df.groupby(['lat','lon'])['tavg'].agg(['max', 'min']).reset_index()
            target_month_df.rename(columns={'max': 'tavg_max', 'min': 'tavg_min'}, inplace=True)
            #print(target_month_df.empty)
            target_month_df.head()

            ref_month = ref[ref['MONTH'] == month]

            #Combine ref data and target data in a single DF
            merged_df = pd.merge(ref_month, target_month_df, on=['lat', 'lon'], how='outer', suffixes=('_df1', '_df2'))
            merged_df.head()
            merged_df['tavg_diff_max'] = merged_df['tavg_max'] - merged_df['tavg']
            merged_df['tavg_diff_min'] = merged_df['tavg_min'] - merged_df['tavg']
            merged_df['tavg_diff_max'] = merged_df['tavg_diff_max'].fillna(0)
            merged_df['tavg_diff_min'] = merged_df['tavg_diff_min'].fillna(0)
            merged_df.head()

            #Set plot params
            fig1, axes1 = plt.subplots(1, 1, figsize=(10, 6),subplot_kw={'projection': ccrs.PlateCarree()})
            ax = axes1
            ax.set_extent([-125.0, -66.93457, 24.396308, 49.384358], crs=ccrs.PlateCarree())  # Set plot extent

            # Add map features like coastime, states
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

            # Add regions data
            for region, bounds in regions.items():
                min_lat, max_lat = bounds["lat"]
                min_lon, max_lon = bounds["lon"]
                rect = Rectangle(
                    (min_lon, min_lat),
                    max_lon - min_lon,
                    max_lat - min_lat,
                    linewidth=0.5,
                    edgecolor="blue",
                    facecolor="none",
                    transform=ccrs.PlateCarree(),
                    label=region
                )
                ax.add_patch(rect)

                center_lon = (min_lon + max_lon) / 2
                center_lat = (min_lat + max_lat) / 2
                ax.text(
                    center_lon, center_lat, region, 
                    transform=ccrs.PlateCarree(), 
                    fontsize=6, color="black", 
                    ha="center", va="center", 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white", alpha=0.2)
                )
            

            ax.set_title(f"{months_str[target_month-1]} - Temperature Difference", fontsize=10)
            cbar_ax1 = fig1.add_axes([0.92, 0.15, 0.02, 0.7])

            fig1.colorbar(sc, cax=cbar_ax1, label="Temperature Difference (°C)")

            # Adjust layout
            fig1.tight_layout(rect=[0, 0, 0.9, 0.95])
            # Master Title
            fig1.suptitle("Avg Temperature Change (Ref - 1960)", fontsize=16)

            #plt.show()
            output_img_name = folder_name + r'\\' + 'TAVG_Diff_' + str(slot_num * duration_gap+start_year+1) + '_' + str(slot_num * duration_gap + start_year + duration_gap) + '.png'
            fig1.savefig(output_img_name)
            print(output_img_name)