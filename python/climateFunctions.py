import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def getMonthLocationOverYears(df, species_id, tracking_month, year_range):
    '''
    Filter the dataframe location of a species in a particular month for all years in the specified range.

    Inputs:
    df             :- full dataframe
    species_id     :- species to track
    tracking_month :- month to filter by
    year_range     :- start and end year (inclusive)

    Output :- filtered dataframe
    '''
    species_data = df[df['SPECIES_ID'] == species_id]
    species_data['MONTH_YEAR'] = pd.to_datetime(species_data['EVENT_YEAR'].astype(str) + '-' + species_data['EVENT_MONTH'].astype(str), errors='coerce')

    migration_data = species_data[['MONTH_YEAR', 'EVENT_MONTH', 'EVENT_YEAR', 'LAT_DD', 'LON_DD']].dropna()
    month_migration = migration_data[migration_data['EVENT_MONTH'] == tracking_month]
    month_migration = month_migration[month_migration['EVENT_YEAR'] >= year_range[0]]
    month_migration = month_migration[month_migration['EVENT_YEAR'] <= year_range[1]]
    return month_migration

def calcCentroidPerYear(month_migration):
    '''
    Calculates the centroid of flock of birds spotted over a month each year.

    Inputs:
    month_migration :- Dataframe filtered ofr location of a species in a particular month for all years in the specified range.

    Output :- filtered dataframe
    '''
    yearly_centroids = (month_migration.groupby('MONTH_YEAR')[['EVENT_YEAR', 'LAT_DD', 'LON_DD']].mean().reset_index().sort_index())
    return yearly_centroids

def plotCentroids(world_shape, yearly_centroids):
    '''
    Plot yearly centroids on a base map.

    Inputs:
    world_shape      :- Base map
    yearly_centroids :- Dataframe of yearly centroids

    Output :- None
    '''
    world = gpd.read_file(f"zip://{world_shape}")
    world = world[world['CONTINENT'] == 'North America']
    ax = world.plot(figsize=(15, 10), color='lightgrey')
    # Superimpose bird sightings
    ax.plot(yearly_centroids['LON_DD'], yearly_centroids['LAT_DD'], color='blue', linewidth=1, markersize=4, label="Path of Movement")
    ax.text(yearly_centroids['LON_DD'][0], yearly_centroids['LAT_DD'][0], "Start", fontsize=10, color='green', fontweight='bold')
    ax.text(yearly_centroids['LON_DD'].iloc[-1], yearly_centroids['LAT_DD'].iloc[-1], "End", fontsize=10, color='red', fontweight='bold')
    for index, centroid in yearly_centroids.iterrows():
        ax.scatter(centroid['LON_DD'], centroid['LAT_DD'], label=f"{int(centroid['EVENT_YEAR'])}", alpha=0.6)

    plt.title(f"Yearly Centroids of Canada Goose Migration ({int(yearly_centroids['EVENT_YEAR'].min())}-{int(yearly_centroids['EVENT_YEAR'].max())})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Year")
    plt.show()

def getFirstSighting(df, species_id, latitudes, year_range, year_step, tracking_month):
    if species_id:
        species_data = df[df['SPECIES_ID'].isin(species_id)]    
    else:
        species_data = df
    species_data['EVENT_DATE'] = pd.to_datetime(species_data['EVENT_DATE'], format='mixed', errors='coerce')    
    species_data['DAY_OF_YEAR'] = species_data['EVENT_DATE'].dt.dayofyear
    species_data['LAT_BAND'] = pd.cut(species_data['LAT_DD'], bins=latitudes, labels=latitudes[:-1])
    species_data = species_data.dropna(subset=['LAT_BAND'])

    year_lst = [y for y in range(year_range[0], year_range[1]+1, year_step)]
    species_data = species_data[species_data['EVENT_YEAR'].isin(year_lst)]
    species_data = species_data[species_data['EVENT_MONTH'] == tracking_month]

    migration_timing = (
        species_data.groupby(['EVENT_YEAR', 'LAT_BAND'])
        .apply(lambda group: group.nsmallest(10, 'DAY_OF_YEAR')['DAY_OF_YEAR'].median())
        .reset_index(name='MEDIAN_DAY_OF_YEAR')
    )

    return migration_timing