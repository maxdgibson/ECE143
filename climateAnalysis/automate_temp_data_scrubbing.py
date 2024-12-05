import pandas as pd
import xarray as xr
import os

def data_scrubber(start_date, end_date, inp_file, year_duration, output_path):
    '''
    Reads the NCEI .nc temperature file (1920-2024 data) and filters it to data between start and end date.

    Input:
    start_date: Starting date
    end_date: Ending date
    inp_file: Input tavg .nc file from NCEI
    year_duration: Intervals of time for each filter file generation (can be set based on system RAM, 5 year data for 8GB RAM)
    output_path: output path containing filtered data in the form of .csv
    '''
    assert(isinstance(start_date, str) and isinstance(end_date, str) and isinstance(inp_file, str))
    assert(isinstance(year_duration, int) and isinstance(output_path, str))
    # Convert to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate 5-year range intervals
    date_range = pd.date_range(start=start_date, end=end_date, freq='5Y')
    intervals = [(date_range[i], date_range[i+1]) for i in range(len(date_range)-1)]
    print(intervals)

    # Output the intervals
    for start, end in intervals:
        output_file_path = output_path + r"\\" + str(start).replace(r" 00:00:00",'') + '_to_' + str(end).replace(r" 00:00:00",'') + '.csv'
        print(f"From {start.date()} to {end.date()}")
        write_header = True
        dataset = xr.open_dataset(inp_file, chunks={"time": 1000000})  # Adjust chunk size as needed

        # Clear the output file if it already exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        for time_chunk in dataset["time"].values:
            time_chunk = pd.to_datetime(time_chunk)

            if (time_chunk > end):
                break
            
            if start <= time_chunk <= end:
                chunk_data = dataset.sel(time=time_chunk)
                df_chunk = chunk_data.to_dataframe().reset_index()
                df_chunk = df_chunk.dropna(subset=['tavg'])

                # Append the filtered data to the CSV file
                if not df_chunk.empty:
                    df_chunk.to_csv(output_file_path, mode='a', index=False, header=write_header)
                    write_header = False

        print(f"Filtered data saved to {output_file_path}")