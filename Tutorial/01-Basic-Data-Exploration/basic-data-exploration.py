import pandas as pd

# Path of the file to read
iowa_file_path = 'Tutorial/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics in next line
print(home_data.describe())