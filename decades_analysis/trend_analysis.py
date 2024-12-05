import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def bird_data_read(path):
    '''
    Read the bird dataset and return it.
    Input: data file path (str)
    Output: DataFrame
    '''
    assert isinstance(path, str), "The file path must be a string."
    bird_csv_path = path
    cols = ['EVENT_YEAR', 'EVENT_MONTH', 'ISO_COUNTRY', 'ISO_SUBDIVISION', 'LAT_DD', 'LON_DD']

    df = pd.read_csv(bird_csv_path, usecols=cols)
    assert not df.empty, "The input file is empty or does not contain required columns."

    print("Columns (label):", df.columns.tolist())
    print(df.head())
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    return df

def bird_data_plot(df):
    '''
    Plot latitude and longitude changes over decades from 1960 to 2020.
    Input: bird dataset (DataFrame)
    '''
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
    assert 'EVENT_YEAR' in df.columns and 'LAT_DD' in df.columns and 'LON_DD' in df.columns, "Required columns are missing."

    df['LAT_DD'] = pd.to_numeric(df['LAT_DD'], errors='coerce')
    df['LON_DD'] = pd.to_numeric(df['LON_DD'], errors='coerce')

    df['EVENT_YEAR'] = pd.to_numeric(df['EVENT_YEAR'], errors='coerce')
    df.dropna(subset=['EVENT_YEAR'], inplace=True)
    df['EVENT_YEAR'] = df['EVENT_YEAR'].astype(int)

    df['EVENT_MONTH'] = pd.to_numeric(df['EVENT_MONTH'], errors='coerce')
    df = df[(df['EVENT_MONTH'] >= 1) & (df['EVENT_MONTH'] <= 12)]
    df['EVENT_MONTH'] = df['EVENT_MONTH'].astype(int)

    df['DECADE'] = (df['EVENT_YEAR'] // 10) * 10

    monthly_lat_data = df.groupby(['DECADE', 'EVENT_MONTH'])['LAT_DD'].mean().reset_index()
    monthly_lon_data = df.groupby(['DECADE', 'EVENT_MONTH'])['LON_DD'].mean().reset_index()

    # Plotting latitude
    plt.figure(figsize=(12, 6))
    for decade in monthly_lat_data['DECADE'].unique():
        subset = monthly_lat_data[monthly_lat_data['DECADE'] == decade]
        plt.plot(subset['EVENT_MONTH'], subset['LAT_DD'], label=f"Decade {decade}")

    plt.title('Average Latitude Over Time by Decade')
    plt.xlabel('Month')
    plt.ylabel('Average Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting longitude
    plt.figure(figsize=(12, 6))
    for decade in monthly_lon_data['DECADE'].unique():
        subset = monthly_lon_data[monthly_lon_data['DECADE'] == decade]
        plt.plot(subset['EVENT_MONTH'], subset['LON_DD'], label=f"Decade {decade}")

    plt.title('Average Longitude Over Time by Decade')
    plt.xlabel('Month')
    plt.ylabel('Average Longitude')
    plt.legend()
    plt.grid(True)
    plt.show()

'''
file_paths = ['./weather_data/grand.csv', './weather_data/sioux.csv', './weather_data/austin.csv', './weather_data/caribo.csv']
station_names = ['Grand', 'Sioux', 'Austin', 'Caribou']
'''
def temperature_plot(file_paths, station_names):
    '''
    Plot the temperature changes in every month of a year over decades for multiple weather stations.

    Input:
        file_paths (list of str): List of file paths to the weather data CSV files.
        station_names (list of str): List of weather station names corresponding to the files.
    '''
    # Ensure the lengths of file paths and station names match
    assert len(file_paths) == len(station_names), "File paths and station names lengths must match."

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Iterate over each file and station name
    for i, (file_path, station_name) in enumerate(zip(file_paths, station_names)):
        # Assert that file_path is a string
        assert isinstance(file_path, str), "File path must be a string."

        df = pd.read_csv(file_path)
        # Assert that required columns are present in the DataFrame
        assert 'DATE' in df.columns and 'TMAX' in df.columns and 'TMIN' in df.columns, \
            "Required columns 'DATE', 'TMAX', and 'TMIN' are missing in the CSV file."


        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Calculate average temperature
        df['TEMPERATURE'] = (df['TMAX'] + df['TMIN']) / 20.0  # Adjusted to a 0.1 scale

        # Extract year and month from the date
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        
        df = df[(df['YEAR'] >= 1960) & (df['YEAR'] <= 2023)]
        df['DECADE'] = (df['YEAR'] // 10) * 10
        monthly_avg_temp = (
            df.groupby(['DECADE', 'MONTH'])['TEMPERATURE']
            .mean()
            .reset_index()
        )

        # Plot data for the current station
        ax = axes[i]
        for decade in monthly_avg_temp['DECADE'].unique():
            subset = monthly_avg_temp[monthly_avg_temp['DECADE'] == decade]
            ax.plot(subset['MONTH'], subset['TEMPERATURE'], label=f"{decade}s")

        ax.set_title(station_name)
        ax.set_xlabel('Month')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

'''
file_paths = ['./weather_data/grand.csv', './weather_data/sioux.csv', './weather_data/austin.csv', './weather_data/caribo.csv']  
station_names = ['Grand', 'Sioux', 'Austin', 'Caribou'] 
'''
def precipitation_plot(file_paths, station_names):
    '''
    Plot the precipitation changes in every month of a year over decades for multiple weather stations.
    Input:
        file_paths (list of str): List of file paths to the weather data CSV files.
        station_names (list of str): List of weather station names corresponding to the files.
    '''
    # Ensure the lengths of file paths and station names match
    assert len(file_paths) == len(station_names), "File paths and station names lengths must match."

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Iterate over each file and station name
    for i, (file_path, station_name) in enumerate(zip(file_paths, station_names)):
        # Assert that file_path is a string
        assert isinstance(file_path, str), "File path must be a string."

        df = pd.read_csv(file_path)

        # Assert that required columns are present in the DataFrame
        assert 'DATE' in df.columns and 'PRCP' in df.columns, \
            "Required columns 'DATE' and 'PRCP' are missing in the CSV file."

        df['DATE'] = pd.to_datetime(df['DATE'])

        df['PRECIPITATION'] = df['PRCP'] / 10.0  # Convert to millimeters (assumes PRCP is in tenths of mm)

        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month

        df = df[(df['YEAR'] >= 1960) & (df['YEAR'] <= 2023)]
        df['DECADE'] = (df['YEAR'] // 10) * 10

        # Calculate monthly average precipitation for each decade
        monthly_avg_precip = (
            df.groupby(['DECADE', 'MONTH'])['PRECIPITATION']
            .mean()
            .reset_index()
        )

        # Plot data for the current station
        ax = axes[i]
        for decade in monthly_avg_precip['DECADE'].unique():
            subset = monthly_avg_precip[monthly_avg_precip['DECADE'] == decade]
            ax.plot(subset['MONTH'], subset['PRECIPITATION'], label=f"{decade}s")

        ax.set_title(f"{station_name} - Average Precipitation")
        ax.set_xlabel('Month')
        ax.set_ylabel('Precipitation (mm)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

'''
we choose Grand and Canada goose here
file_path = "./weather_data/grand.csv"
df, monthly_avg_temp, monthly_avg_precip = relation()
'''
def relation(file_path):
    '''
    Analyze the relationship between temperature, precipitation, and special weather events over decades.

    Input:
        file_paths (str): file path to the bird dataset.

    Output:
        df (DataFrame): Processed DataFrame containing the original data with additional computed columns.
        monthly_avg_temp (DataFrame): Monthly average temperature by decade and month.
        monthly_avg_precip (DataFrame): Monthly average precipitation by decade and month.
    '''
    assert isinstance(file_path, str), "File path must be a string."
    df = pd.read_csv(file_path)

    required_columns = ['DATE', 'TMAX', 'TMIN', 'PRCP', 'AWND']
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' is missing in the dataset."

    df['DATE'] = pd.to_datetime(df['DATE'])

    # temperature (average of TMAX and TMIN, divided by 20)
    df['TEMPERATURE'] = (df['TMAX'] + df['TMIN']) / 20.0

    # precipitation in millimeters
    df['PRECIPITATION'] = df['PRCP'] / 10.0

    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month

    df = df[(df['YEAR'] >= 1960) & (df['YEAR'] <= 2023)]
    df['AWND'] = df['AWND'] / 10.0
    df['DECADE'] = (df['YEAR'] // 10) * 10

    # Define columns representing extreme weather events
    extreme_weather = ['WT01', 'WT04', 'WT05', 'WT10', 'WT11', 'WT15', 'WT17', 'WT18']
    extreme_weather_columns = [col for col in df.columns if col in extreme_weather]

    # the number of special weather events for each record
    df['SPECIAL_EVENTS'] = df[extreme_weather_columns].notnull().sum(axis=1)

    # monthly average temperature
    monthly_avg_temp = (
        df.groupby(['DECADE', 'MONTH'])['TEMPERATURE']
        .mean()
        .reset_index()
    )

    # monthly average precipitation
    monthly_avg_precip = (
        df.groupby(['DECADE', 'MONTH'])['PRECIPITATION']
        .mean()
        .reset_index()
    )

    return df, monthly_avg_temp, monthly_avg_precip

'''
bird_csv_path = './NABBP_2023_grp_02.csv'
'''
def preci_vs_location(bird_csv_path, df, monthly_avg_precip):
    '''
    Analyze the relationship between precipitation and bird migration location (latitude and longitude).

    Input:
        df (DataFrame): Original weather DataFrame, processed with relevant columns.
        monthly_avg_precip (DataFrame): Monthly average precipitation by decade and month.

    Output:
        None (The function displays scatter plots and regression lines, and prints regression coefficients and intercepts).
    '''

    assert isinstance(bird_csv_path, str), "Bird CSV file path must be a string."

    cols = ['EVENT_YEAR', 'EVENT_MONTH', 'ISO_COUNTRY', 'ISO_SUBDIVISION', 'LAT_DD', 'LON_DD']

    bird_df = pd.read_csv(bird_csv_path, usecols=cols)

    for col in cols:
        assert col in bird_df.columns, f"Required column '{col}' is missing in bird dataset."

    # Convert latitude and longitude to numeric
    bird_df['LAT_DD'] = pd.to_numeric(bird_df['LAT_DD'], errors='coerce')
    bird_df['LON_DD'] = pd.to_numeric(bird_df['LON_DD'], errors='coerce')

    # Convert year and month to numeric and clean up missing values
    bird_df['EVENT_YEAR'] = pd.to_numeric(bird_df['EVENT_YEAR'], errors='coerce')
    bird_df.dropna(subset=['EVENT_YEAR'], inplace=True)
    bird_df['EVENT_YEAR'] = bird_df['EVENT_YEAR'].astype(int)

    bird_df['EVENT_MONTH'] = pd.to_numeric(bird_df['EVENT_MONTH'], errors='coerce')
    bird_df = bird_df[(bird_df['EVENT_MONTH'] >= 1) & (bird_df['EVENT_MONTH'] <= 12)]
    bird_df['EVENT_MONTH'] = bird_df['EVENT_MONTH'].astype(int)

    bird_df = bird_df.dropna(subset=['EVENT_YEAR', 'EVENT_MONTH'])
    bird_df['DECADE'] = (bird_df['EVENT_YEAR'] // 10) * 10

    # Group by decade and month to calculate average latitude and longitude
    monthly_avg_lat = bird_df.groupby(['DECADE', 'EVENT_MONTH'])['LAT_DD'].mean().reset_index()
    monthly_avg_lon = bird_df.groupby(['DECADE', 'EVENT_MONTH'])['LON_DD'].mean().reset_index()

    df['DECADE'] = df['DECADE'].astype(int)
    df['MONTH'] = df['MONTH'].astype(int)
    bird_df['DECADE'] = bird_df['DECADE'].astype(int)
    bird_df['EVENT_MONTH'] = bird_df['EVENT_MONTH'].astype(int)

    merged_lat = pd.merge(monthly_avg_precip, monthly_avg_lat, left_on=['DECADE', 'MONTH'], right_on=['DECADE', 'EVENT_MONTH'], how='inner')

    # Regression for latitude vs precipitation
    X_lat = merged_lat[['PRECIPITATION']]
    y_lat = merged_lat['LAT_DD']
    model_lat = LinearRegression()
    model_lat.fit(X_lat, y_lat)
    y_pred_lat = model_lat.predict(X_lat)

    merged_lon = pd.merge(monthly_avg_precip, monthly_avg_lon, left_on=['DECADE', 'MONTH'], right_on=['DECADE', 'EVENT_MONTH'], how='inner')

    # Regression for longitude vs precipitation
    X_lon = merged_lon[['PRECIPITATION']]
    y_lon = merged_lon['LON_DD']
    model_lon = LinearRegression()
    model_lon.fit(X_lon, y_lon)
    y_pred_lon = model_lon.predict(X_lon)

    # Plot latitude vs precipitation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(X_lat, y_lat, label='Latitude Data points', color='blue', marker='p')
    ax1.plot(X_lat, y_pred_lat, color='red', label='Latitude Regression Line', linewidth=4)
    ax1.set_xlabel('Precipitation (mm)')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Relationship Between Precipitation (mm) and Latitude')
    ax1.legend()
    ax1.grid(True)

    # Plot longitude vs precipitation
    ax2.scatter(X_lon, y_lon, label='Longitude Data points', color='green', marker='x')
    ax2.plot(X_lon, y_pred_lon, color='orange', label='Longitude Regression Line', linewidth=4)
    ax2.set_xlabel('Precipitation (mm)')
    ax2.set_ylabel('Longitude')
    ax2.set_title('Relationship Between Precipitation (mm) and Longitude')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print regression coefficients and intercepts
    print(f'Latitude Regression Coefficient: {model_lat.coef_[0]}')
    print(f'Latitude Intercept: {model_lat.intercept_}')
    print(f'Longitude Regression Coefficient: {model_lon.coef_[0]}')
    print(f'Longitude Intercept: {model_lon.intercept_}')



