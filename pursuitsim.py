import numpy as np
import pandas as pd
import random
import math
import time
import threading
#import matplotlib.pyplot as plt
from scipy.stats import vonmises
import dearpygui.dearpygui as dpg
import json

def gen_angular(kappa, angular_min, angular_max, angular_peak_shift, size=100000):
    # Convert deg to rad
    angular_min_rad = np.radians(angular_min)
    angular_max_rad = np.radians(angular_max)
    
    # Generate samples from Von Mises distribution
    angular_samples = vonmises.rvs(kappa, size=size)
    
    # Calculate the range width
    angular_range_width = angular_max_rad - angular_min_rad
    
    # Rescale Von Mises samples to fit within [0, range_width]
    samples_rescaled = (angular_samples + np.pi) * (angular_range_width / (2 * np.pi))
    
    # Shift samples to the desired range [angular_min, angular_max]
    samples_shifted = samples_rescaled - angular_peak_shift + angular_min_rad
    
    # Wrap around to fit within [angular_min, angular_max]
    angular_dist = (samples_shifted - angular_min_rad) % angular_range_width + angular_min_rad
    
    return np.degrees(angular_dist)

def gen_linear(mu, sigma, linear_min, linear_max, size=100000):

    # Generate samples from a Gaussian distribution
    linear_samples = np.random.normal(mu, sigma, size)
    
    # Calculate the range width
    linear_range_width = linear_max - linear_min
    
    # Wrap samples to fit within [linear_min, linear_max]
    linear_dist = (linear_samples - linear_min) % linear_range_width + linear_min
    
    return linear_dist

def gen_gamma(shape, scale, peak, min_x, max_x, size=100000):
    # Generate samples from a Gamma distribution
    gamma_samples = np.random.gamma(shape, scale, size)
    
    # Transform samples to adjust peak position
    mean_sample = shape * scale  # Gamma distribution mean
    shift = peak - mean_sample  # Calculate the shift required
    samples_shifted = gamma_samples + shift
    
    # Filter out samples that are outside the range [min_x, max_x]
    gamma_dist = samples_shifted[(samples_shifted >= min_x) & (samples_shifted <= max_x)]
    
    return gamma_dist

# Function to run a single trial
def run_trial(d, a, l):
    x_pos = random.uniform(-0.5, 0.8) # Generate random starting locations
    y_pos = random.uniform(-0.5, 0.8)
    
    col_x, col_y = [], [] # Initialize lists for X and Y DataFrame columns
    delta_x, delta_y, angular_velocity = 0, 0, 0 # Initialize location modifiers for x and y, and theta from AV
    
    for t in range(d):
        if t == 0:
            pass
        elif t == 1 or t % 10 == 0: # Every 10 loops, change the direction of the dot by changing the modifier
            angular_velocity += np.random.choice(a) - (angular_max / 2) # Adds a new sample to total angle
            linear_velocity = 1 + (np.random.choice(l) / linear_max) # LV provides 1.XX multiplier (max of 2)
            delta_x = math.cos(math.radians(angular_velocity)) * linear_velocity # Modifiers for x and y
            delta_y = math.sin(math.radians(angular_velocity)) * linear_velocity

        x_pos += delta_x # Every loop, add the modifier to x and y
        y_pos += delta_y
        col_x.append(x_pos) # Add x and y to the DataFrame lists
        col_y.append(y_pos)

    return pd.DataFrame({'X': col_x, 'Y': col_y}) # Return finished DataFrame

# Parameters
HZ = 60
a_dist = 'Wrapped'
l_dist = 'Gaussian'
angular_min = -90
angular_max = 90
angular_range = angular_max + angular_min
linear_min = 0
linear_max = 100
linear_range = linear_max + linear_min
sigma = (linear_max - linear_min) * 0.215
peak = 32
kappa = 1.0
a_shape = 3.0
a_scale = 1.0
l_shape = 5.0
l_scale = 3.0
trials = 50
traj = {}
a = None
l = None
w_size = 526
h_size = 267.5
wp_size = 250
hp_size = 250

def export_to_csv():
    # Create an empty DataFrame to hold all trials
    combined_df = pd.DataFrame()

    # Loop through all trials and concatenate them into one DataFrame
    for trial_index in range(1, trials + 1):
        if trial_index in traj:
            trial_df = traj[trial_index]
            trial_df['Trial'] = trial_index  # Add a column to identify the trial
            combined_df = pd.concat([combined_df, trial_df], ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('trial_results.csv', index=False)
    print("CSV exported successfully!")

def display_results(index):
    result_text = f"Trial {index}:\n{traj[index]}"
    
    if dpg.does_item_exist("Simulation Results"):
        dpg.delete_item("Simulation Results")
    
    with dpg.window(label="Simulation Results", width=204.5, height=515, tag="Simulation Results", no_title_bar=True, no_scrollbar=True):
        with dpg.table(header_row=False):
            dpg.add_table_column()
            with dpg.table_row():                
                dpg.add_text("     Simulation Results")
        dpg.add_text(result_text)

    if dpg.does_item_exist('Export Results'):
        dpg.delete_item('Export Results')

    with dpg.window(label="Export Results", width=204.5, height=268, tag="Export Results", no_title_bar=True):
        dpg.set_item_pos("Export Results", (1001, 515))

        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()       
            with dpg.table_row():
                dpg.add_text('') 
            with dpg.table_row():
                dpg.add_text('') 
            with dpg.table_row():
                dpg.add_text('') 
            with dpg.table_row():
                dpg.add_text('') 
            with dpg.table_row():
                dpg.add_text('') 
            with dpg.table_row():
                dpg.add_spacer()                 
                dpg.add_button(label="Export", width=-1, callback=export_to_csv)       
                dpg.add_spacer()  

    # Set the position of the results window
    dpg.set_item_pos("Simulation Results", (1001, 0))

def exit_program(sender, app_data):
    dpg.stop_dearpygui()

# Update functions for min/max values
def update_angular_min(value):
    global angular_min, angular_range
    angular_min = value
    angular_range = angular_max + angular_min
    print(f"Updated angular_min to {angular_min}, angular_range to {angular_range}")

def update_angular_max(value):
    global angular_max, angular_range
    angular_max = value
    angular_range = angular_max + angular_min
    print(f"Updated angular_max to {angular_max}, angular_range to {angular_range}")

def update_linear_min(value):
    global linear_min, linear_range, sigma
    linear_min = value
    linear_range = linear_max + linear_min
    sigma = (linear_max - linear_min) * 0.215
    print(f"Updated linear_min to {linear_min}, linear_range to {linear_range}, sigma to {sigma}")

def update_linear_max(value):
    global linear_max, linear_range, sigma
    linear_max = value
    linear_range = linear_max + linear_min
    sigma = (linear_max - linear_min) * 0.215
    print(f"Updated linear_max to {linear_max}, linear_range to {linear_range}, sigma to {sigma}")

def create_gui():
    dpg.create_context()

    with dpg.window(label="Pursuit Parameters", width=475, height=782.5, tag='main', no_title_bar=True):
        
        with dpg.table(header_row=False):
            # Define the columns
            dpg.add_table_column()  
            dpg.add_table_column()  
            dpg.add_table_column()

            with dpg.table_row():
                dpg.add_spacer()
                dpg.add_text("Pursuit Parameters")
                dpg.add_spacer()

        with dpg.table(header_row=False):
            # Define the columns
            dpg.add_table_column()  
            dpg.add_table_column()  
            #dpg.add_table_column()  

            with dpg.table_row():
                dpg.add_text("Adjust Distributions") 

            # Dropdowns for distributions
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Angular Distribution")
                dpg.add_combo(items=['Gaussian', 'Wrapped', 'Skewed'], width=200, default_value=a_dist, callback=lambda s, a: globals().update({'a_dist': a}))
                dpg.add_spacer()

            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Linear Distribution")
                dpg.add_combo(items=['Gaussian', 'Wrapped', 'Skewed'], width=200, default_value=l_dist, callback=lambda s, a: globals().update({'l_dist': a}))
                dpg.add_spacer()
            
            # Angular Min
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Angular Min")
                dpg.add_input_float(default_value=angular_min, width=200, callback=lambda s, a: update_angular_min(a))
                dpg.add_spacer()

            # Angular Max
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Angular Max")
                dpg.add_input_float(default_value=angular_max, width=200, callback=lambda s, a: update_angular_max(a))
                dpg.add_spacer()
            
            # Linear Min
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Linear Min")
                dpg.add_input_float(default_value=linear_min, width=200, callback=lambda s, a: update_linear_min(a))
                dpg.add_spacer()
            
            # Linear Max
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Linear Max")
                dpg.add_input_float(default_value=linear_max, width=200, callback=lambda s, a: update_linear_max(a))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text('') 

            with dpg.table_row():
                dpg.add_text('Experimental') 

            # Peak
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Gamma Peak")
                dpg.add_input_float(default_value=peak, width=200, callback=lambda s, a: globals().update({'peak': a}))
                dpg.add_spacer()

            # Kappa
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Kappa")
                dpg.add_input_float(default_value=kappa, width=200, callback=lambda s, a: globals().update({'kappa': a}))
                dpg.add_spacer()

            # a Shape
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Angular Shape")
                dpg.add_input_float(default_value=a_shape, width=200, callback=lambda s, a: globals().update({'a_shape': a}))
                dpg.add_spacer()

            # a Scale
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Angular Scale")
                dpg.add_input_float(default_value=a_scale, width=200, callback=lambda s, a: globals().update({'a_scale': a}))
                dpg.add_spacer()

            # l Shape
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Linear Shape")
                dpg.add_input_float(default_value=l_shape, width=200, callback=lambda s, a: globals().update({'l_shape': a}))
                dpg.add_spacer()

            # l Scale
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Linear Scale")
                dpg.add_input_float(default_value=l_scale, width=200, callback=lambda s, a: globals().update({'l_scale': a}))
                dpg.add_spacer()

            # w size
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Plot Window Width")
                dpg.add_input_float(default_value=w_size, width=200, callback=lambda s, a: globals().update({'w_size': a}))
                dpg.add_spacer()

            # h size
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Plot Window Height")
                dpg.add_input_float(default_value=h_size, width=200, callback=lambda s, a: globals().update({'h_size': a}))
                dpg.add_spacer()

            # wp size
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Plot Width")
                dpg.add_input_float(default_value=wp_size, width=200, callback=lambda s, a: globals().update({'wp_size': a}))
                dpg.add_spacer()

            # hp size
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Plot Height")
                dpg.add_input_float(default_value=hp_size, width=200, callback=lambda s, a: globals().update({'hp_size': a}))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text('') 

            with dpg.table_row():
                dpg.add_text('Playback Settings') 

            # Playback Speed
            with dpg.table_row():
                #dpg.add_spacer()
                dpg.add_text("    Playback Speed")
                dpg.add_input_float(default_value=speed_multiplier, width=200, callback=lambda s, a: globals().update({'speed_multiplier': a}))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text("")

        # Buttons to run simulation and exit
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()       
            with dpg.table_row():
                dpg.add_spacer()                 
                dpg.add_button(label="Run Simulation", width=-1, callback=pursuit_sim)      
                dpg.add_spacer()  
            with dpg.table_row():
                dpg.add_spacer() 
            with dpg.table_row():
                dpg.add_spacer()   
                dpg.add_button(label="Run All", width=-1, callback=pursuit_sim_all)
                dpg.add_spacer()
            with dpg.table_row():
                dpg.add_spacer() 
            with dpg.table_row():
                dpg.add_spacer()   
                dpg.add_button(label="Exit", width=-1, callback=exit_program)
                dpg.add_spacer()

        #print(dpg.get_item_configuration('main'))

    with dpg.window(label="Plot Distributions", width=w_size, height=h_size, tag="Plot Distributions", no_title_bar=True):
        dpg.set_item_pos("Plot Distributions", (475, 515))

    # with dpg.window(label='Trajectory Map', tag='Trajectory Plot'):
    #     with dpg.tab_bar(tag='tab_bar'):
    #         dpg.add_tab(label='Trajectory Plot')
    #         dpg.add_tab(label='Laser')
    #         #dpg.set_item_pos('tab_bar', (475.0))

    with dpg.window(label="Trajectory Map", width=526.5, height=515, tag="Trajectory Plot", no_title_bar=True):
        dpg.set_item_pos("Trajectory Plot", (475, 0))

    with dpg.window(label="Export Results", width=204.5, height=268, tag="Export Results", no_title_bar=True):
        dpg.set_item_pos("Export Results", (1001, 515))

    with dpg.window(label="Simulation Results", width=204.5, height=515, tag="Simulation Results", no_title_bar=True):
        dpg.set_item_pos("Simulation Results", (1001, 0))
        # with dpg.table(header_row=False):
        #     dpg.add_table_column()
        #     with dpg.table_row():                
        #         dpg.add_text("     Simulation Results")

    # Enable docking
    dpg.configure_app(docking=True, docking_space=True)

    dpg.create_viewport(title='Pursuit Simulator', width=1206, height=782)
    dpg.setup_dearpygui()
    #dpg.maximize_viewport()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# gen_angular(kappa, angular_min, angular_max, angular_peak_shift, size=100000)
# def gen_gamma(shape, scale, peak, min_x, max_x, size=100000)

# Plot variables
current_plot_index = 0
current_frame = 0
timer_interval = 100
speed_multiplier = 1.0
laser_width = 500
laser_height = 450
x_pos = 0
y_pos = 0

def show_plot(index):
    global current_frame, is_playing
    if is_playing:  # Stop playback if already running
        is_playing = False
    
    current_plot_index = index
    current_frame = 0  # Reset the frame counter

    if index < 1 or index > len(traj):
        return  # Out of bounds check

    df = traj[index]

    # Calculate the trial duration in seconds
    trial_duration = len(df) / HZ

    # Update the displayed DataFrame for the current trial
    display_results(index)

    # Check if the plot window exists and delete it
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")

    # Create a new plot window with the tab bar
    with dpg.window(label="Trajectory Map", width=526, height=515, tag="Trajectory Plot", no_title_bar=True, no_scrollbar=True):
        with dpg.tab_bar(tag='tab_bar'):
            with dpg.tab(label="Trajectory Plot"):
                plot_id = dpg.add_plot(label=f'Trajectory Map - Trial {index} ({trial_duration:.2f} s)', width=510, height=450)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='X Coordinate', parent=plot_id)
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='Y Coordinate', parent=plot_id)

                # Store the line series in a way that we can update it later
                trajectory_series = dpg.add_line_series(df['X'].tolist(), df['Y'].tolist(), label="Trajectory", parent=y_axis, tag="Trajectory Series")

            with dpg.tab(label="Laser"):
                with dpg.drawlist(width=500, height=450, tag="laser_drawlist"):
                    # Draw the initial green circle at the origin
                    min_x, max_x = df['X'].min(), df['X'].max()
                    min_y, max_y = df['Y'].min(), df['Y'].max()
                    translate_x = (x_pos - min_x) / (max_x - min_x) * laser_width
                    translate_y = laser_height - (y_pos - min_y) / (max_y - min_y) * laser_height
                    dpg.draw_circle(center=(translate_x, translate_y), radius=10, color=(0, 255, 0, 255), tag="Laser Circle", fill=(0, 255, 0))

        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()       
            with dpg.table_row():
                # Navigation buttons
                dpg.add_button(label="Previous", width=-1, callback=lambda: show_plot(index - 1) if index > 1 else None)
                dpg.add_button(label="Play", width=-1, callback=lambda: start_playback(df))
                dpg.add_button(label="Next", width=-1, callback=lambda: show_plot(index + 1) if index < len(traj) else None)

            
    dpg.set_item_pos("Trajectory Plot", (475, 0))

def show_combination_plot(trial_index):
    global current_frame, is_playing

    if is_playing:  # Stop playback if already running
        is_playing = False

    current_frame = 0  # Reset the frame counter

    # Clear the previous plot if it exists
    if dpg.does_item_exist("All Combinations Plot"):
        dpg.delete_item("All Combinations Plot")

    # Create a new plot window for the combined trajectories
    with dpg.window(label="All Combination Trajectories", width=526, height=515, tag="All Combinations Plot", no_title_bar=True, no_scrollbar=True):
        plot_id = dpg.add_plot(label=f'Trajectories for All Combinations - Trial {trial_index}', width=510, height=450)
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='X Coordinate', parent=plot_id)
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='Y Coordinate', parent=plot_id)

        # Loop through all combinations and plot their trajectories for this trial
        for i, ((angular_type, linear_type), trial_data) in enumerate(traj_all_combinations.items()):
            df = trial_data[trial_index]
            # Use a unique tag for each line series
            dpg.add_line_series(df['X'].tolist(), df['Y'].tolist(), label=f"{angular_type} + {linear_type}", parent=y_axis, tag=f"Trajectory Series {i}")

        # Navigation buttons for trials
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()       
            with dpg.table_row():
                dpg.add_button(label="Previous", width=-1, callback=lambda: show_combination_plot(trial_index - 1) if trial_index > 1 else None)
                dpg.add_button(label="Play", width=-1, callback=lambda: start_combined_playback(trial_index))
                dpg.add_button(label="Next", width=-1, callback=lambda: show_combination_plot(trial_index + 1) if trial_index < 50 else None)

    dpg.set_item_pos("All Combinations Plot", (475, 0))

def update_combination_plot(trial_index):
    global current_frame, is_playing

    if current_frame < len(trial_data) and is_playing:
        # Loop through all combinations
        for i, ((angular_type, linear_type), trial_data) in enumerate(traj_all_combinations.items()):
            df = trial_data[trial_index]  # Access the DataFrame for the current trial
            x_data = dpg.get_value(f"Trajectory Series {i}")[0]
            y_data = dpg.get_value(f"Trajectory Series {i}")[1]
            
            x_pos = df['X'].iloc[current_frame]  # Access X coordinate
            y_pos = df['Y'].iloc[current_frame]  # Access Y coordinate

            # Update the line series for each combination
            dpg.set_value(f"Trajectory Series {i}", (x_data + [x_pos], y_data + [y_pos]))

            # Calculate min and max for translating to the laser plot
            min_x, max_x = df['X'].min(), df['X'].max()
            min_y, max_y = df['Y'].min(), df['Y'].max()
            translate_x = (x_pos - min_x) / (max_x - min_x) * laser_width
            translate_y = laser_height - (y_pos - min_y) / (max_y - min_y) * laser_height

            # Update the laser circle for each combination
            dpg.configure_item(f"Laser Circle {i}", center=(translate_x, translate_y))

        current_frame += 1

        # Schedule the next update
        threading.Timer(timer_interval / 1000, update_combination_plot, args=(trial_index,)).start()

def playback_thread(df):
    global current_frame
    start_time = time.time()
    
    while current_frame < len(df):
        elapsed = time.time() - start_time
        if elapsed >= timer_interval / 1000:  # Check if the time interval has passed
            update_plot(df)
            start_time = time.time()  # Reset the timer
        time.sleep(0.01)  # Sleep briefly to avoid high CPU usage

# Set timer interval based on HZ
timer_interval = (1000 / HZ) / speed_multiplier  # Interval in milliseconds

is_playing = False  # Flag to track if playback is active

def start_playback(df):
    global current_frame, is_playing, timer_interval
    current_frame = 0
    is_playing = True
    
    # Recalculate the timer interval
    timer_interval = (1000 / HZ) / speed_multiplier
    
    # Clear previous trajectory
    dpg.set_value("Trajectory Series", ([], []))
    dpg.set_value("Laser Circle", ([], []))
    
    # Start the playback
    update_plot(df)

def start_combined_playback(trial_index):
    global current_frame, is_playing, timer_interval
    is_playing = True
    current_frame = 0  # Reset the frame counter
    
    # Iterate through all combinations and update the plot
    for (angular_type, linear_type), trial_data in traj_all_combinations.items():
        df = trial_data[trial_index]  # Access the DataFrame for the current trial
        update_combination_plot(df)  # Pass the df directly

def update_plot(df):
    global current_frame, is_playing

    if current_frame < len(df) and is_playing:
        # Append the current point to the line series
        x_data = dpg.get_value("Trajectory Series")[0]
        y_data = dpg.get_value("Trajectory Series")[1]
        
        x_pos = df['X'].iloc[current_frame]
        y_pos = df['Y'].iloc[current_frame]
        
        # Update the line series
        dpg.set_value("Trajectory Series", (x_data + [x_pos], y_data + [y_pos]))

        min_x, max_x = df['X'].min(), df['X'].max()
        min_y, max_y = df['Y'].min(), df['Y'].max()
        translate_x = (x_pos - min_x) / (max_x - min_x) * laser_width
        translate_y = laser_height - (y_pos - min_y) / (max_y - min_y) * laser_height

        # Update the laser circle position
        dpg.configure_item("Laser Circle", center=(translate_x, translate_y))

        current_frame += 1

        # Schedule the next update
        threading.Timer(timer_interval / 1000, update_plot, args=(df,)).start()

# def play_plot(df):
#     global is_playing
#     is_playing = True
#     update_plot(df)

# def pause_plot():
#     global is_playing
#     is_playing = False

def pursuit_sim():
    global traj, a, l

    if a_dist == 'Gaussian':
        angular_peak_shift = 0
        a = gen_angular(kappa, angular_min, angular_max, angular_peak_shift)
    elif a_dist == 'Wrapped':
        angular_peak_shift = angular_max - (angular_range / 2)
        a = gen_angular(kappa, angular_min, angular_max, np.radians(angular_peak_shift))
    elif a_dist == 'Skewed':
        a = gen_gamma(a_shape, a_scale, peak, angular_min, angular_max)
        
    if l_dist == 'Gaussian':
        l = gen_linear(linear_range / 2, sigma, linear_min, linear_max)
    elif l_dist == 'Wrapped':
        l = gen_linear(linear_max, sigma, linear_min, linear_max)
    elif l_dist == 'Skewed':
        l = gen_gamma(l_shape, l_scale, peak, linear_min, linear_max)
        
    # Create dictionary of n trials
    #trials = 50
    traj = {} # Initialize dictionary of DataFrames
    
    # Duration uses operations rather than time. Ex. at 120Hz, 120 = 1 sec
    for trial in range(1, trials + 1):
        d = np.random.randint(2 * HZ, (8 * HZ) + 1) # Min = 2s, Max = 8s
        traj[trial] = run_trial(d, a, l)
    
    # Show full DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    #a = a
    #l = l
    
    display_results(1)
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")
        
    with dpg.window(label="Trajectory Map", width=526, height=515, tag="Trajectory Plot", no_title_bar=True):
        with dpg.tab_bar(tag='tab_bar'):
            dpg.add_tab(label='Trajectory Plot')
            dpg.add_tab(label='Laser')
            dpg.add_tab(label='All Combinations')
        dpg.set_item_pos("Trajectory Plot", (475, 0))
    show_plot(1)
    show_plots()
    
    return a, l

def pursuit_sim_all():
    global traj_all_combinations

    dist_types = ['Gaussian', 'Wrapped', 'Skewed']
    
    # Initialize dictionary to hold trials for each combination
    traj_all_combinations = {comb: {} for comb in [(a, l) for a in dist_types for l in dist_types if a != 'Skewed']}

    for angular_type in dist_types:
        for linear_type in dist_types:
            # Skip Angular Skewed combinations
            if angular_type == 'Skewed':
                continue
            
            if angular_type == 'Gaussian':
                a = gen_angular(kappa, angular_min, angular_max, 0)
            elif angular_type == 'Wrapped':
                angular_peak_shift = angular_max - (angular_range / 2)
                a = gen_angular(kappa, angular_min, angular_max, np.radians(angular_peak_shift))
            elif angular_type == 'Skewed':
                # This case won't be executed due to the continue statement
                continue

            if linear_type == 'Gaussian':
                l = gen_linear(linear_range / 2, sigma, linear_min, linear_max)
            elif linear_type == 'Wrapped':
                l = gen_linear(linear_max, sigma, linear_min, linear_max)
            elif linear_type == 'Skewed':
                l = gen_gamma(l_shape, l_scale, peak, linear_min, linear_max)

            # Run trials for this combination
            for trial in range(1, 51):  # Assuming 50 trials
                d = np.random.randint(2 * HZ, (8 * HZ) + 1)  # Min = 2s, Max = 8s
                traj_all_combinations[(angular_type, linear_type)][trial] = run_trial(d, a, l)

    # Show the combined results
    show_combination_plot(1)

    
    # Function to create and show plots
def show_plots():
    global a, l  # Access global variables

    # Check if the plot window exists and delete it
    if dpg.does_item_exist("Plot Distributions"):
        dpg.delete_item("Plot Distributions")

    # Angular data for plotting
    hist_a, edges_a = np.histogram(a, bins=100, density=True)
    x_a = (edges_a[:-1] + edges_a[1:]) / 2  # Centers of bins

    # Linear data for plotting
    hist_l, edges_l = np.histogram(l, bins=100, density=True)
    x_l = (edges_l[:-1] + edges_l[1:]) / 2  # Centers of bins

    with dpg.window(label="Plot Distributions", width=w_size, height=h_size, tag="Plot Distributions", no_title_bar=True):
        with dpg.group(horizontal=True):  # Group to arrange plots horizontally
            # Angular Distribution Plot
            with dpg.plot(label="Angular Distribution", height=hp_size, width=wp_size):
                dpg.add_plot_axis(dpg.mvXAxis, label="Angle (degrees)")
                dpg.add_plot_axis(dpg.mvYAxis, label="Density", tag="y_axis_a")
                #dpg.add_spacer(width=5)
                dpg.add_bar_series(x=list(x_a), y=list(hist_a), label="Angular Density", parent="y_axis_a")

        #with dpg.window(label="Linear Plot", width=w_size, height=h_size, tag="Linear Plot"):
            # Linear Distribution Plot
            with dpg.plot(label="Linear Distribution", height=hp_size, width=wp_size):
                dpg.add_plot_axis(dpg.mvXAxis, label="Linear Value")
                dpg.add_plot_axis(dpg.mvYAxis, label="Density", tag="y_axis_l")
                dpg.add_bar_series(x=list(x_l), y=list(hist_l), label="Linear Density", parent="y_axis_l")

    # Set the position of the plots window
    dpg.set_item_pos("Plot Distributions", (475, 515)) 
    #dpg.set_item_pos("Linear Plot", (865, 0)) 

# Run the GUI
create_gui()
    #print(traj) # Print all DataFrames (can disable)

#plt.hist(a, bins=100, density=True, alpha=0.6, color='g')

