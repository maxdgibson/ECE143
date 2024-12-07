# ECE143
Final Project Repository

Data Sets:
Bird Data
https://www.sciencebase.gov/catalog/item/653fa806d34ee4b6e05bc57d

Climate Data
https://www.ncei.noaa.gov/pub/data/cirs/climgrid/nclimgrid_tavg.nc
Note: this dataset has NOT been pushed to github as the file size is too large. Please download this file to perform climate trend analyses.

Climate Analysis Test Code:
1. Generate Temp Heat Maps (Sample Test 1 - /test/ folder)
```
    temp_heat_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse)
```



2. Generate Bird Movement Maps overlaid on temp heatmaps (Sample Test 2 - /test/ folder)
```
    bird_mov_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse, bird_data_file, bounds)
```
