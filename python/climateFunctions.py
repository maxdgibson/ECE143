import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import calendar

def getMonthLocationOverYears(df, species_id, tracking_month, year_range, year_step):
    '''
    Filter the dataframe location of a species in a particular month for all years in the specified range.

    Inputs:
    df             :- full dataframe
    species_id     :- list of species to track
    tracking_month :- month to filter by
    year_range     :- start and end year (inclusive)
    year_step      :- year increment step size

    Output :- filtered dataframe
    '''

    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert all(col in df.columns for col in ['SPECIES_ID', 'EVENT_YEAR', 'EVENT_MONTH', 'LAT_DD', 'LON_DD']), "df must contain columns: SPECIES_ID, EVENT_YEAR, EVENT_MONTH, LAT_DD, LON_DD."
    assert isinstance(species_id, list) and all(isinstance(i, int) for i in species_id), "species_id must be a list of integers."
    assert isinstance(tracking_month, int) and 1 <= tracking_month <= 12, "tracking_month must be an integer between 1 and 12."
    assert isinstance(year_range, list) and len(year_range) == 2 and all(isinstance(y, int) for y in year_range), "year_range must be a list of two integers."
    assert isinstance(year_step, int) and year_step > 0, "year_step must be a positive integer."

    # Select all required species
    if species_id:
        species_data = df[df['SPECIES_ID'].isin(species_id)]    
    else:
        species_data = df

    species_data['MONTH_YEAR'] = pd.to_datetime(species_data['EVENT_YEAR'].astype(str) + '-' + species_data['EVENT_MONTH'].astype(str), errors='coerce')

    # Remove invalid data rows
    migration_data = species_data[['MONTH_YEAR', 'EVENT_MONTH', 'EVENT_YEAR', 'LAT_DD', 'LON_DD']].dropna()
    # Select the required month
    month_migration = migration_data[migration_data['EVENT_MONTH'] == tracking_month]

    # Filter requried years at regular time steps
    year_lst = [y for y in range(year_range[0], year_range[1]+1, year_step)]
    month_migration = month_migration[month_migration['EVENT_YEAR'].isin(year_lst)]

    return month_migration


def calcCentroidPerMonthYear(month_migration):
    '''
    Calculates the centroid of flock of birds spotted over a month each year.

    Inputs:
    month_migration :- Dataframe filtered for location of a species in a particular month for all years in the specified range.

    Output :- filtered dataframe
    '''

    assert isinstance(month_migration, pd.DataFrame), "month_migration must be a pandas DataFrame."
    assert all(col in month_migration.columns for col in ['MONTH_YEAR', 'EVENT_YEAR', 'EVENT_MONTH', 'LAT_DD', 'LON_DD']), "month_migration must contain columns: MONTH_YEAR, EVENT_YEAR, EVENT_MONTH, LAT_DD, LON_DD."

    # Find the mean (centroid) location of all birds spotted in a month, grouped by years
    monthly_centroids = (month_migration.groupby('MONTH_YEAR')[['EVENT_YEAR', 'EVENT_MONTH', 'LAT_DD', 'LON_DD']].mean().reset_index().sort_index())
    return monthly_centroids


def plotYearlyCentroids(world_shape, species, yearly_centroids):
    '''
    Plot yearly centroids on a base map.

    Inputs:
    world_shape      :- Base map
    species          :- Species name
    yearly_centroids :- Dataframe of yearly centroids

    Output :- None
    '''

    assert isinstance(world_shape, str), "world_shape must be a string representing the path to a zip file."
    assert isinstance(species, str), "species must be a string."
    assert isinstance(yearly_centroids, pd.DataFrame), "yearly_centroids must be a pandas DataFrame."
    assert all(col in yearly_centroids.columns for col in ['LAT_DD', 'LON_DD', 'EVENT_YEAR', 'EVENT_MONTH']), "yearly_centroids must contain columns: LAT_DD, LON_DD, EVENT_YEAR, EVENT_MONTH."

    # Plot the North American map
    world = gpd.read_file(f"zip://{world_shape}")
    world = world[world['CONTINENT'] == 'North America']
    ax = world.plot(figsize=(15, 10), color='lightgrey')
    
    # Superimpose bird sightings
    ax.text(yearly_centroids['LON_DD'][0], yearly_centroids['LAT_DD'][0], "Start", fontsize=10, color='green', fontweight='bold')
    ax.text(yearly_centroids['LON_DD'].iloc[-1], yearly_centroids['LAT_DD'].iloc[-1], "End", fontsize=10, color='red', fontweight='bold')
    for index, centroid in yearly_centroids.iterrows():
        ax.scatter(centroid['LON_DD'], centroid['LAT_DD'], label=f"{int(centroid['EVENT_YEAR'])}", alpha=0.6)

    start_year = int(yearly_centroids['EVENT_YEAR'].min())
    end_year   = int(yearly_centroids['EVENT_YEAR'].max())
    month      = calendar.month_name[int(yearly_centroids['EVENT_MONTH'][0])]
    plt.title(f"Yearly Centroids of Bird Migration: {species} \n{month} ({start_year}-{end_year})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Year")
    plt.show()


def plotMonthlyCentroids(world_shape, species, monthly_centroids):
    '''
    Plot yearly centroids on a base map.

    Inputs:
    world_shape      :- Base map
    species          :- Species name
    monthly_centroids :- Dataframe of yearly centroids

    Output :- None
    '''

    assert isinstance(world_shape, str), "world_shape must be a string representing the path to a zip file."
    assert isinstance(species, str), "species must be a string."
    assert isinstance(monthly_centroids, pd.DataFrame), "monthly_centroids must be a pandas DataFrame."
    assert all(col in monthly_centroids.columns for col in ['LAT_DD', 'LON_DD', 'EVENT_YEAR', 'EVENT_MONTH']), "monthly_centroids must contain columns: LAT_DD, LON_DD, EVENT_YEAR, EVENT_MONTH."

    # Plot the North American map
    world = gpd.read_file(f"zip://{world_shape}")
    world = world[world['CONTINENT'] == 'North America']
    ax = world.plot(figsize=(15, 10), color='lightgrey')

    # Superimpose bird sightings
    mid = len(monthly_centroids['LON_DD']) // 2
    ax.text(monthly_centroids['LON_DD'][0], monthly_centroids['LAT_DD'][0], f"{calendar.month_name[int(monthly_centroids['EVENT_MONTH'][0])]}", fontsize=10, color='green', fontweight='bold')
    ax.text(monthly_centroids['LON_DD'][mid], monthly_centroids['LAT_DD'][mid], f"{calendar.month_name[int(monthly_centroids['EVENT_MONTH'][mid])]}", fontsize=10, color='red', fontweight='bold')
    for index, centroid in monthly_centroids.iterrows():
        ax.scatter(centroid['LON_DD'], centroid['LAT_DD'], label=f"{calendar.month_name[int(centroid['EVENT_MONTH'])]}", alpha=0.6)

    plt.title(f"Monthly Centroids of Bird Migration: {species} \n(Year: {int(monthly_centroids['EVENT_YEAR'].max())})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Year")
    plt.show()


def getFirstSighting(df, species_id, latitudes, year_range, year_step, tracking_month):
    '''
    This function groups first sightings of a species between specified latitudes in the month
    under consideration.
    Median of the first 10 sightings is used to reduce noise and avoid abberant sightings.

    Inputs:
    df             :- full dataframe
    species_id     :- species to track
    latitude       :- latitude ranges to group first sightings
    year_range     :- start and end year (inclusive)
    year_step      :- year increment step size
    tracking_month :- month to filter by

    Output :- filtered dataframe
    '''

    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert all(col in df.columns for col in ['SPECIES_ID', 'EVENT_DATE', 'LAT_DD', 'EVENT_YEAR', 'EVENT_MONTH']), "df must contain columns: SPECIES_ID, EVENT_DATE, LAT_DD, EVENT_YEAR, EVENT_MONTH."
    assert isinstance(species_id, list) and all(isinstance(i, int) for i in species_id), "species_id must be a list of integers."
    assert isinstance(latitudes, list) and all(isinstance(lat, (int, float)) for lat in latitudes), "latitudes must be a list of numbers."
    assert isinstance(year_range, list) and len(year_range) == 2 and all(isinstance(y, int) for y in year_range), "year_range must be a list of two integers."
    assert isinstance(year_step, int) and year_step > 0, "year_step must be a positive integer."
    assert isinstance(tracking_month, int) and 1 <= tracking_month <= 12, "tracking_month must be an integer between 1 and 12."

    # Select all required species
    if species_id:
        species_data = df[df['SPECIES_ID'].isin(species_id)]    
    else:
        species_data = df

    # Select data by latitude ranges
    species_data['EVENT_DATE'] = pd.to_datetime(species_data['EVENT_DATE'], format='mixed', errors='coerce')    
    species_data['DAY_OF_YEAR'] = species_data['EVENT_DATE'].dt.dayofyear
    species_data['LAT_BAND'] = pd.cut(species_data['LAT_DD'], bins=latitudes, labels=latitudes[:-1])
    species_data = species_data.dropna(subset=['LAT_BAND'])

    # Select a month for year year under observation
    year_lst = [y for y in range(year_range[0], year_range[1]+1, year_step)]
    species_data = species_data[species_data['EVENT_YEAR'].isin(year_lst)]
    species_data = species_data[species_data['EVENT_MONTH'] == tracking_month]

    migration_timing = (
        species_data.groupby(['EVENT_YEAR', 'LAT_BAND'])
        .apply(lambda group: group.nsmallest(10, 'DAY_OF_YEAR')['DAY_OF_YEAR'].median())
        .reset_index(name='MEDIAN_DAY_OF_YEAR')
    )

    return migration_timing