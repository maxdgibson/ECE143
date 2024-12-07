from climateAnalysis import bird_mov_map_gen, temp_heat_map_gen, stich_images_as_gif

def main():
    #Input Parameters
    
    #Path containing input tavg csv file scrubbed using automate_temp_data_scrubbing.py file
    inp_path = r'<path_to_dataset_organized_as_csv_file_using_automate_temp_data_scrubbing.py'
    start_year = 1960
    end_year = 2020
    duration_gap = 5
    months_to_analyse = [4,5,6]
    bird_data_file = r'pkl_file_containing_bird_centroid_movement_data' #Sample provided in /datasets/migration/ path
    region_bounds = [-94, -78, 35, 50]

    print("Executing bird map gen test")
    #Generate Temp Heat Maps (Outputs will be generated in the input folder)
    temp_heat_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse)
    print("Done Test 1")

    print("Executing bird map gen test")
    #Generate bird movement (Outputs will be generated in the input folder):
    bird_mov_map_gen(inp_path, start_year, end_year, duration_gap, months_to_analyse, bird_data_file, region_bounds)
    print("Done Test 2")


if __name__ == "__main__":
    main()