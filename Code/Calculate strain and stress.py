# -*- coding: utf-8 -*-
"""stress strain

"""

#@title  Function Definitions

from sklearn import linear_model, datasets


def ransac_fit(X, strain_array, subset = False, plot_fit = True):
    '''
    Linear Fit with RANSAC
    subset: interval e.g. [0.1, 0.8] is grabbing data from 10-80 %
            i.e. discarding edges
    '''

    if subset != False:
        array_indeces = [int(i_ * strain_array.size) for i_ in subset]
        X = X[array_indeces[0] : array_indeces[1]]
        strain_array = strain_array[array_indeces[0] : array_indeces[1]]


    X = X.reshape(-1,1)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X, strain_array)



    line_y_ransac = model_ransac.predict(X)

    if plot_fit:
        plt.plot(X, line_y_ransac,
                linestyle = 'dashed', color = [0.4,0.4,0.4,0.15], linewidth = 2, zorder = -10)  # color is grey
    return model_ransac


def get_strain(distance_array):

    strain_array = 0.5*(distance_array**2 / distance_array[0]**2) ; strain_array = strain_array - 1
    return strain_array


def eucledian_distance_2d(x1, y1, x2, y2):
    distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return distance

def get_distance(filename, get_csv_columns_ = True):
    '''
    specific to what napri points layers .csv returns
    will return the two columns for x and y position
    '''
    points_df = pd.read_csv(filename)


    len_array = int(points_df['axis-0'].max())

    distance_array = np.zeros(len_array)
    X = np.array([ i for i in range(len_array)])

    for i in range(len_array):
        points_index = np.array(points_df[points_df['axis-0'] == i])
        x1 = points_index[0,2]
        x2 = points_index[1,2]
        y1 = points_index[0,3]
        y2 = points_index[1,3]

        distance_array[i] = np.array(eucledian_distance_2d(x1,y1,x2,y2))

    return distance_array


def get_deflection_from_strain(strain, initial_length = 1e-3, cantilever_speed = 0.003 * 1e-3, seconds_between_frames = 15):
    '''
    input: strain - as measured by the separation of the two filter paper patches

    output: the beam deflection (to measure the force)

    ### All units in SI

    keyword argument inputs:
    initial_length := (of paper separation) (2 mm)
    cantilever_speed = 0.003 mm/s (~ strain rate of actuator)
    seconds_between_frames := 20 seconds

    '''
    strain = np.array(strain)
    real_time = seconds_between_frames * np.arange(0,strain.size) # strain.size := number of frames
    paper_displacement = (strain) * initial_length
    deflection = cantilever_speed * real_time - paper_displacement
    return deflection


def get_cantilever_force_from_deflection(deflection, stiffness_k =  (1.67 * 1e3) * 1e-6 / 1e-2):
    '''
    input: deflection(s) of the beam - can be a single number, or a list of numbers
           or a numpy array - will be converted to a numpy array

    output: the stress corresponding to the deflections

    mg to N =  9.8067E-6 /



    '''
    return np.array(deflection) * stiffness_k

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import xarray as xr

import warnings
warnings.filterwarnings("ignore")


import os
from contextlib import contextmanager

@contextmanager
def temporary_cd(path):
    """
    Temporarily change the working directory.
    """
    original_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_path)

# # Usage
# FOLDER_PATH = '/path/to/your/directory'
# with temporary_cd(FOLDER_PATH):
#     # Your code that needs to run in FOLDER_PATH





PROJECT_FOLDER = r'../DATA/'

EXPERIMENT_FOLDER = r'experiment/'

EXPERIMENT_FOLDER = PROJECT_FOLDER + EXPERIMENT_FOLDER


filenames_ = os.listdir(EXPERIMENT_FOLDER)


### Read file name
FILE_NAME = EXPERIMENT_FOLDER + filenames_[0]



print(FILE_NAME)
df = pd.read_csv(FILE_NAME)

# Convert the pandas DataFrame to an xarray Dataset
ds = df.to_xarray()

# Display the contents of the xarray Dataset
ds

#@markdown Rename as necessary for the dimension (column) names to have meaning



# Function to find the data variable with less than 20% unique values (i.e. the Z coordinate)
def find_low_unique_var(data):
    for key, values in data.items():
        unique_values = np.unique(values)
        if len(unique_values) / len(values) < 0.2:
            return key.split('-')[-1]
    return None # menaing no column matches the criterion

# Rename the data variables

def annotate_names_to_xarray(ds):

    try:
        ### LC data
        ### X and Y positions are last two dims
        ds = ds.rename({
        f'Y': 'Y_POS',
        f'X': 'X_POS',
        f'Frame': 'TIME'
                    })

        return ds
    except:
        try:
            ### LC data
           ### X and Y positions are last two dims
            ds = ds.rename({
            f'Y': 'Y_POS',
            f'X': 'X_POS',
            f'Slice': 'TIME'
                    })

            return ds
        except:
            pass

    n_vars =  len(ds.data_vars)

    plausible_axes = [ str(int(i)) for i in range(n_vars) ]


    ### X and Y positions are last two dims
    ds = ds.rename({
        f'axis-{n_vars - 2}': 'Y_POS',
        f'axis-{n_vars - 1}': 'X_POS',
                   })


    # remove the last two from the possible axes
    plausible_axes = plausible_axes[:-2]

    # remove the Z axis if it exists
    try:
        plausible_axes.remove(find_low_unique_var(ds))
    except ValueError:
        pass

    # the only remaining axis must be time
        ### X and Y positions are last two dims
    ds = ds.rename({
        f'axis-{plausible_axes[0]}': 'TIME',
                   })

    return ds


# Display the contents of the xarray Dataset
ds = annotate_names_to_xarray(ds)
ds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(ds):

    # Extract the X and Y coordinates
    x = ds['X_POS'].values
    y = ds['Y_POS'].values

    # Ensure there are exactly two values for each coordinate at each time point
    if any(len(ds[coord].values) != 2 for coord in ['X_POS', 'Y_POS']):
        # Return NaN if there are not two values for each coordinate
        distance = np.nan
    else:
        # Compute the Euclidean distance
        distance = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    # distance_xr = xr.Dataset({'Distance' : distance , 'TIME' : ds['TIME'].values[0]})
    # Create the Dataset with 'Distance' as a data variable and 'TIME' as a coordinate
    distance_xr = xr.Dataset(
        data_vars={'Distance': distance},
        coords={'TIME': ds['TIME'].values[0]}  # Associating with the first time value
    )

    # print(distance_xr)


    return distance_xr

# Function to find the data variable with less than 20% unique values
def find_low_unique_var(data):
    for key, values in data.items():
        unique_values = np.unique(values)
        if len(unique_values) / len(values) < 0.2:
            return key.split('-')[-1]
    return None

def compute_strain_and_rolling_avg(ds):

    ## Annotate names to the columns of the  xarray
    ds = annotate_names_to_xarray(ds)

    distance_ds = calculate_distance(ds)
    # Group by TIME and apply the distance calculation
    distance_ds = ds.groupby('TIME').map(calculate_distance)

    # Define the initial length as the distance at the first Time
    initial_length = distance_ds.Distance.isel(TIME = 0)
    #  ....................      ^ choose Distance variable
    # ................                     ^ isel TIME = 0 (choose form the coordinate)

    # Calculate the strain using the formula
    distance_ds['Strain'] = (distance_ds['Distance']-initial_length)/initial_length
    distance_ds['Lagrangian_Strain'] = 0.5*((distance_ds['Distance']**2-initial_length**2)/initial_length**2)
    # Compute the rolling average of the strain with a window size of 2 and min_periods=1
    distance_ds['Rolling_Avg'] = distance_ds['Strain'].rolling(TIME=2, min_periods=1, center=True).mean().dropna("TIME")
    distance_ds['Lagrangian_Rolling_Avg'] = distance_ds['Lagrangian_Strain'].rolling(TIME=2, min_periods=1, center=True).mean().dropna("TIME")

    # print(distance_ds)

    return distance_ds

def get_files_with_pattern(filename_list, regex_pattern):
    matched_files = []
    for filename in filename_list:
        if re.search(regex_pattern, filename):
            matched_files.append(filename)
    return matched_files

def calculate_rolling_avg_strain_2_series(sample_num, id):
    ### Load the CSV data into DataSets (xarrays)
    df_endo_1 = pd.read_csv(f'sample{sample_num}_{id}_endo1.csv').to_xarray()
    df_endo_2 = pd.read_csv(f'sample{sample_num}_{id}_endo2.csv').to_xarray()

    # Compute strain and rolling average for both endo and meso
    ### AND convert them back to dataframes
    df_avg_endo_1 = compute_strain_and_rolling_avg(df_endo_1).to_dataframe().reset_index()
    df_avg_endo_2 = compute_strain_and_rolling_avg(df_endo_2).to_dataframe().reset_index()

    # Take average
    # Merging the dataframes on 'TIME' with an outer join
    merged_df = pd.merge(df_avg_endo_1, df_avg_endo_2, on='TIME', how='outer', suffixes=('_1', '_2'))

    # Calculating the average 'Rolling_Avg', using 'mean' over axis=1 (row-wise)
    # 'mean' automatically skips NaN values, effectively using the available value if one is missing
    merged_df['Rolling_Avg'] = merged_df[['Rolling_Avg_1', 'Rolling_Avg_2']].mean(axis=1)
    merged_df['Lagrangian_Rolling_Avg']=merged_df[['Lagrangian_Rolling_Avg_1','Lagrangian_Rolling_Avg_2']].mean(axis=1)

    # Creating the new dataframe with required columns
    df = merged_df[['TIME', 'Rolling_Avg', 'Lagrangian_Rolling_Avg']]

    return df

## Plot strain vs time




