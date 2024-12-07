# ECE143 Team 2
Final Project Repository
cliMATE: Climate and Migration Analysis for Tracking Ecosystems

Bird migration is a critical ecological process that has important functions such as pollination and pest control. As climate change alters seasonal patterns and habitats, migratory birds face unprecedented challenges, potentially leading to shifts in migration timing and routes. Understanding these relationships is essential for predicting potential disruptions in ecological processes and aiding conservation efforts to mitigate these effects.

Our proposed solution is to analyze climate data and bird migration patterns to draw correlations on how migratory patterns would evolve. We plan to use Pandas to manage data, numpy to perform linear algebra and matplotlib for data visualization. Supporting libraries to extract and read datasets are also used.

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

Migration Analysis Test Code:
1. Visualize typical migration path of a bird species in an year (test\test_normalMigrationPath.ipynb)

2. Visualize shifts in migration centroids over large time-steps (test\test_geoSpatial.ipynb)

3. Visualize changes in arrival time at a geospatial location along the migration path (test\test_delayAnalysis.ipynb)

Correlation Analyses
1. Correlation analyses between temperature changes and migratory path shifts (correlations\correlation_between_temperature_and_center.ipynb)
