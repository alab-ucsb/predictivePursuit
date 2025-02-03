import numpy as np
import pandas as pd
import random
import math
import time
import threading
from scipy.stats import vonmises
import dearpygui.dearpygui as dpg
import json
from datetime import datetime, timedelta

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

def run_trial(d, a, l):
    global scale_x, scale_y, x_range, y_range

    while True:  # Keep repeating until a valid starting point is achieved
        # Randomly initialize X and Y within the specified ranges
        target_x = 0
        target_y = 0

        # Set initial position to the target
        x_pos = target_x
        y_pos = target_y

        # Initialize lists for storing trajectory
        col_x, col_y = [], []
        delta_x, delta_y, angular_velocity = 0, 0, 0

        for t in range(d):
            if t == 0:
                pass
            elif t == 1 or t % t_mod == 0:
                angular_velocity += np.random.choice(a) - (angular_max / 2)
                linear_velocity = np.random.choice(l) / t_mod
                delta_x = math.cos(math.radians(angular_velocity)) * linear_velocity
                delta_y = math.sin(math.radians(angular_velocity)) * linear_velocity

            x_pos += delta_x
            y_pos += delta_y

            # Wrap around logic for x-axis
            if x_pos > 29.21:
                x_pos = -29.21
            elif x_pos < -29.21:
                x_pos = 29.21
            
            # Wrap around logic for y-axis
            if y_pos > 31.4325:
                y_pos = -31.4325
            elif y_pos < -31.4325:
                y_pos = 31.4325
            
            col_x.append(x_pos)
            col_y.append(y_pos)

        # Create DataFrame
        df = pd.DataFrame({'X': col_x, 'Y': col_y})
        
        print(f"Generated Trial with starting position ({df['X'].iloc[0]}, {df['Y'].iloc[0]})")
        return df

# Parameters
HZ = 30
t_mod = 5
a_dist = 'Wrapped'
l_dist = 'Skewed'
angular_min = 60
angular_max = 180
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
time = 20
scale_x = 0.7
scale_y = 0.7
start_min_x = -scale_x
start_max_x = scale_x
start_min_y = -0.2
start_max_y = 0.3
x_range = (start_min_x, start_max_x)
y_range = (start_min_y, start_max_y)
adjusted_time = 10
default_date = {
        "year": datetime.now().year,
        "month": f"{datetime.now().month:02d}",
        "day": f"{datetime.now().day:02d}"
    }

def export_to_csv(filename):

    print(f"Filename type: {type(filename)}")
    print(f"Filename: {filename}")
    combined_df = pd.DataFrame()

    for trial_index in range(1, int(trials) + 1):
        if trial_index in traj:
            trial_df = traj[trial_index]
            trial_df['Trial'] = trial_index
            trial_df['HZ'] = HZ
            trial_df['Duration (s)'] = len(trial_df) / HZ

            combined_df = pd.concat([combined_df, trial_df], ignore_index=True)

    combined_df.to_csv(filename, index=False)
    print(f"CSV exported successfully as {filename}!")

def generate_daily_trajectories(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        filename = f"Trajectories/traj_{current_date.month}-{current_date.day}-{current_date.strftime('%y')}.csv"
        pursuit_sim()  # Run the simulation
        export_to_csv(filename)  # Export the result to a CSV with the date-specific filename
        current_date += timedelta(days=1)

def export_csvs():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 7, 1)
    generate_daily_trajectories(start_date, end_date)

def display_results(index):
    result_text = f"Trial {index}:\n{traj[index].iloc[:, :2]}"

        # Check if the plot window exists and delete it
    if dpg.does_item_exist("Trajectory Info"):
        dpg.delete_item("Trajectory Info")

    # Create a new plot window with the tab bar
    with dpg.window(label="Trajectory Info", width=204.5, height=515, tag="Trajectory Info", no_title_bar=True, no_scrollbar=True, no_move=True):
        with dpg.tab_bar(tag='trajectory_info'):
            with dpg.tab(label="Editor"):
                # Buttons for editing the current trajectory
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_button(label="Regenerate", width=-1, callback=lambda: regen_trajectory(index))
                    with dpg.table_row():
                        dpg.add_button(label="Flip X", width=-1, callback=lambda: flip_x(index))
                    with dpg.table_row():
                        dpg.add_button(label="Flip Y", width=-1, callback=lambda: flip_y(index))
                dpg.add_text('Adjust Time')
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("Duration")
                        dpg.add_input_float(default_value=time, step=0, width=-1, callback=lambda s, a: globals().update({'time': a}))
                    with dpg.table_row():
                        dpg.add_input_float(default_value=adjusted_time, step=0, width=-1, callback=lambda s, a: globals().update({'adjusted_time': a}))
                        dpg.add_button(label="Extend", width=-1, callback=lambda: extend_trajectory(index))
                    with dpg.table_row():
                        dpg.add_input_float(default_value=adjusted_time, step=0, width=-1, callback=lambda s, a: globals().update({'adjusted_time': a}))
                        dpg.add_button(label="Trim", width=-1, callback=lambda: trim_trajectory(index))

            with dpg.tab(label="Coordinates"):
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    with dpg.table_row():                
                        dpg.add_text("     Simulation Results")
                dpg.add_text(result_text)

            with dpg.tab(label="Norm"):
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    with dpg.table_row():                
                        dpg.add_text("     Normalized Results")
                dpg.add_text(traj[index].iloc[:, 2:4])
    # Set the position of the results window
    dpg.set_item_pos("Trajectory Info", (1001, 0))

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

    with dpg.window(label="Pursuit Parameters", width=475, height=782.5, tag='main', no_title_bar=True, no_move=True):
        
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

            with dpg.table_row():
                dpg.add_text("Adjust Distributions") 

            # Dropdowns for distributions
            with dpg.table_row():
                dpg.add_text("    Angular Distribution")
                dpg.add_combo(items=['Gaussian', 'Wrapped', 'Skewed'], width=200, default_value=a_dist, callback=lambda s, a: globals().update({'a_dist': a}))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text("    Linear Distribution")
                dpg.add_combo(items=['Gaussian', 'Wrapped', 'Skewed'], width=200, default_value=l_dist, callback=lambda s, a: globals().update({'l_dist': a}))
                dpg.add_spacer()
            
            # Angular Min
            with dpg.table_row():
                dpg.add_text("    Angular Min")
                dpg.add_input_float(default_value=angular_min, width=200, callback=lambda s, a: update_angular_min(a))
                dpg.add_spacer()

            # Angular Max
            with dpg.table_row():
                dpg.add_text("    Angular Max")
                dpg.add_input_float(default_value=angular_max, width=200, callback=lambda s, a: update_angular_max(a))
                dpg.add_spacer()
            
            # Linear Min
            with dpg.table_row():
                dpg.add_text("    Linear Min")
                dpg.add_input_float(default_value=linear_min, width=200, callback=lambda s, a: update_linear_min(a))
                dpg.add_spacer()
            
            # Linear Max
            with dpg.table_row():
                dpg.add_text("    Linear Max")
                dpg.add_input_float(default_value=linear_max, width=200, callback=lambda s, a: update_linear_max(a))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text('') 

            with dpg.table_row():
                dpg.add_text('Experimental') 

            # Peak
            with dpg.table_row():
                dpg.add_text("    Gamma Peak")
                dpg.add_input_float(default_value=peak, width=200, callback=lambda s, a: globals().update({'peak': a}))
                dpg.add_spacer()

            # Kappa
            with dpg.table_row():
                dpg.add_text("    Kappa")
                dpg.add_input_float(default_value=kappa, width=200, callback=lambda s, a: globals().update({'kappa': a}))
                dpg.add_spacer()

            # a Shape
            with dpg.table_row():
                dpg.add_text("    Angular Shape")
                dpg.add_input_float(default_value=a_shape, width=200, callback=lambda s, a: globals().update({'a_shape': a}))
                dpg.add_spacer()

            # a Scale
            with dpg.table_row():
                dpg.add_text("    Angular Scale")
                dpg.add_input_float(default_value=a_scale, width=200, callback=lambda s, a: globals().update({'a_scale': a}))
                dpg.add_spacer()

            # l Shape
            with dpg.table_row():
                dpg.add_text("    Linear Shape")
                dpg.add_input_float(default_value=l_shape, width=200, callback=lambda s, a: globals().update({'l_shape': a}))
                dpg.add_spacer()

            # l Scale
            with dpg.table_row():
                dpg.add_text("    Linear Scale")
                dpg.add_input_float(default_value=l_scale, width=200, callback=lambda s, a: globals().update({'l_scale': a}))
                dpg.add_spacer()

            # Playback Speed
            with dpg.table_row():
                dpg.add_text("    Playback Speed")
                dpg.add_input_float(default_value=speed_multiplier, width=200, callback=lambda s, a: globals().update({'speed_multiplier': a}))
                dpg.add_spacer()

            with dpg.table_row():
                dpg.add_text('') 

            with dpg.table_row():
                dpg.add_text('Trajectory Parameters') 

            # Operations Per Second (HZ)
            with dpg.table_row():
                dpg.add_text("    Operations Per Second (HZ)")
                dpg.add_input_float(default_value=HZ, width=200, callback=lambda s, a: globals().update({'HZ': a}))
                dpg.add_spacer()

            # Operations Per Trajectory
            with dpg.table_row():
                dpg.add_text("    Operations Per Trajectory")
                dpg.add_input_float(default_value=t_mod, width=200, callback=lambda s, a: globals().update({'t_mod': a}))
                dpg.add_spacer()

            # # Time
            # with dpg.table_row():
            #     dpg.add_text("    Time")
            #     dpg.add_input_float(default_value=time, width=200, callback=lambda s, a: globals().update({'time': a}))
            #     dpg.add_spacer()
            
            # Trials
            with dpg.table_row():
                dpg.add_text("    Trials")
                dpg.add_input_float(default_value=trials, width=200, callback=lambda s, a: globals().update({'trials': a}))
                dpg.add_spacer()

            # Scale X
            with dpg.table_row():
                dpg.add_text("    Scale X")
                dpg.add_input_float(default_value=scale_x, width=200, callback=lambda s, a: globals().update({'scale_x': a}))
                dpg.add_spacer()

            # Scale Y
            with dpg.table_row():
                dpg.add_text("    Scale Y")
                dpg.add_input_float(default_value=scale_y, width=200, callback=lambda s, a: globals().update({'scale_y': a}))
                dpg.add_spacer()

            # Min X Start Range 
            with dpg.table_row():
                dpg.add_text("    Min X Start Position")
                dpg.add_input_float(default_value=start_min_x, width=200, callback=lambda s, a: globals().update({'start_min_x': a}))
                dpg.add_spacer()

            # Max X Start Range 
            with dpg.table_row():
                dpg.add_text("    Max X Start Position")
                dpg.add_input_float(default_value=start_max_x, width=200, callback=lambda s, a: globals().update({'start_max_x': a}))
                dpg.add_spacer()

            # Min Y Start Range 
            with dpg.table_row():
                dpg.add_text("    Min Y Start Position")
                dpg.add_input_float(default_value=start_min_y, width=200, callback=lambda s, a: globals().update({'start_min_y': a}))
                dpg.add_spacer()

            # Max Y Start Range 
            with dpg.table_row():
                dpg.add_text("    Max Y Start Position")
                dpg.add_input_float(default_value=start_max_y, width=200, callback=lambda s, a: globals().update({'start_max_y': a}))
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
            # with dpg.table_row():
            #     dpg.add_spacer()   
            #     dpg.add_button(label="Run All", width=-1, callback=pursuit_sim_all)
            #     dpg.add_spacer()
            # with dpg.table_row():
            #     dpg.add_spacer() 
            with dpg.table_row():
                dpg.add_spacer()   
                dpg.add_button(label="Exit", width=-1, callback=exit_program)
                dpg.add_spacer()

        #print(dpg.get_item_configuration('main'))

    with dpg.window(label="Plot Distributions", width=w_size, height=h_size, tag="Plot Distributions", no_title_bar=True, no_move=True):
        dpg.set_item_pos("Plot Distributions", (475, 515))

    with dpg.window(label="Trajectory Map", width=526.5, height=515, tag="Trajectory Plot", no_title_bar=True, no_move=True):
        dpg.set_item_pos("Trajectory Plot", (475, 0))

    with dpg.window(label="Export Results", width=204.5, height=268, tag="Export Results", no_title_bar=True, no_move=True):
        dpg.set_item_pos("Export Results", (1001, 515))
        current_date = datetime.now()
        default_date = {
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day
        }

        dpg.add_date_picker(label="Date", tag="Date", callback=import_trajectory_csv, default_value={'month_day': default_date['day'], 'year': default_date['year'] - 1900, 'month': default_date['month'] - 1})
        #dpg.set_item_pos("Date", (10, 0)) # callback causes error, check import_trajectory_csv

        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                dpg.add_text('')
            with dpg.table_row():
                dpg.add_spacer()                 
                dpg.add_button(label="Export", width=-1, callback=lambda: export_to_csv(f"Trajectories/trial_results_{datetime.now().strftime('%m-%d-%y')}.csv"))       
                dpg.add_spacer()  
            # with dpg.table_row():
            #     dpg.add_text('') 
            # with dpg.table_row():
            #     dpg.add_spacer() 
            #     dpg.add_spacer()                
            #     # dpg.add_button(label="Export All", width=-1, callback=export_csvs)       
            #     dpg.add_spacer()
            with dpg.table_row():
                dpg.add_spacer()
                dpg.add_button(label="Import", width=-1, callback=import_csv)       
                dpg.add_spacer()  
            # with dpg.table_row():
            #     dpg.add_spacer()                 
            #     dpg.add_button(label="Import", width=-1, callback=import_trajectory_csv)       
            #     dpg.add_spacer()
            #with dpg.table_row():
            #    dpg.add_spacer()
        #dpg.add_date_picker(label="Date")
            #    dpg.add_spacer()

    with dpg.window(label="Simulation Results", width=204.5, height=515, tag="Simulation Results", no_title_bar=True, no_move=True):
        dpg.set_item_pos("Simulation Results", (1001, 0))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Enable docking
    dpg.configure_app(docking=True, docking_space=True)

    dpg.create_viewport(title='Pursuit Simulator', width=1206, height=782)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    pursuit_sim()
    #import_trajectory_csv()
    dpg.start_dearpygui()
    dpg.destroy_context()

# Plot variables
current_plot_index = 0
current_frame = 0
timer_interval = 100
speed_multiplier = 1.0
laser_width = 500
laser_height = 450
x_pos = 0
y_pos = 0

# Function to import trajectories from a CSV file
def import_trajectory_csv():
    #Imports a CSV file and populates the `traj` dictionary with trial data
    global traj
    traj.clear()
    
    selected_month = dpg.get_value('Date')['month'] + 1
    selected_day = dpg.get_value('Date')['month_day']
    selected_year = dpg.get_value('Date')['year'] - 100
    filepath = f"Trajectories/traj_{selected_month}-{selected_day}-{selected_year}.csv" #/Users/jarodbussey/Downloads/predictivePursuit-main/Trajectories/traj_{selected_month}-{selected_day}-{selected_year}.csv"

    # Load the CSV and group by 'Trial' to organize data
    df = pd.read_csv(filepath)
    for trial, data in df.groupby("Trial"):
        traj[trial] = data[['X', 'Y']].reset_index(drop=True)

    # Check if the plot window exists and delete it
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")

    # Show the plot for the first trial in the CSV data
    if traj:
        show_plot(1)

    print(f"Successfully imported CSV data from {filepath}")

def import_csv():
    #Imports a CSV file and populates the `traj` dictionary with trial data
    global traj, default_date
    traj.clear()

    filepath = f"Trajectories/trial_results_{datetime.now().strftime('%m-%d-%y')}.csv"

    # Load the CSV and group by 'Trial' to organize data
    df = pd.read_csv(filepath)
    for trial, data in df.groupby("Trial"):
        traj[trial] = data[['X', 'Y', 'Norm X', 'Norm Y']].reset_index(drop=True)

    # Check if the plot window exists and delete it
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")

    # Show the plot for the first trial in the CSV data
    if traj:
        show_plot(1)

    print(f"Successfully imported CSV data from {filepath}")

# Show trajectory plot
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

    # Display results for the current trial
    display_results(index)

    # Check if the plot window exists and delete it
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")

    # Create a new plot window with the tab bar
    with dpg.window(label="Trajectory Map", width=526, height=515, tag="Trajectory Plot", no_title_bar=True, no_scrollbar=True, no_move=True):
        with dpg.tab_bar(tag='tab_bar'):
            with dpg.tab(label="Trajectory Plot"):
                with dpg.plot(label=f'Trajectory Map - Trial {index} ({trial_duration:.2f} s)', width=510, height=450, tag='traj_plot'):
                    dpg.add_plot_axis(dpg.mvXAxis, label='x_axis', tag='traj_x')
                    dpg.add_plot_axis(dpg.mvYAxis, label='y_axis', tag='traj_y')
                    dpg.set_axis_limits('traj_x', -30, 30)
                    dpg.set_axis_limits('traj_y', -32, 32)
                    # Plot trajectory
                    dpg.add_line_series(df['X'].tolist(), df['Y'].tolist(), parent='traj_y', label="Trajectory", tag="Trajectory Series")

            with dpg.tab(label="Laser"):
                with dpg.drawlist(width=500, height=450, tag="laser_drawlist"):
                    # Draw the initial green circle at the origin
                    min_x, max_x = df['X'].min(), df['X'].max()
                    min_y, max_y = df['Y'].min(), df['Y'].max()
                    translate_x = (x_pos - min_x) / (max_x - min_x) * laser_width
                    translate_y = laser_height - (y_pos - min_y) / (max_y - min_y) * laser_height
                    dpg.draw_circle(center=(translate_x, translate_y), radius=10, color=(0, 255, 0, 255), tag="Laser Circle", fill=(0, 255, 0))

        # Navigation buttons for navigating trials
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()     
            #dpg.add_table_column()  
            with dpg.table_row():
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
    with dpg.window(label="All Combination Trajectories", width=526, height=515, tag="All Combinations Plot", no_title_bar=True, no_scrollbar=True, no_move=True):
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
                dpg.add_button(label="Next", width=-1, callback=lambda: show_combination_plot(trial_index + 1) if trial_index < trials else None)

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

def pursuit_sim():
    global traj, a, l

    # Load the CSV file with durations
    durations_file = "durations.csv"  # Path to your CSV file
    durations_df = pd.read_csv(durations_file, header=0)  # Assumes the file has one column of durations
    duration_values = durations_df.iloc[:, 0].values  # Extracts the first column as a list or array
    
    # Ensure the number of rows matches the number of trials
    if len(duration_values) < int(trials):
        raise ValueError("The number of rows in the CSV is less than the number of trials.")

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
      
    print("Durations loaded")  
    # Create dictionary of n trials
    #trials = 50
    traj = {} # Initialize dictionary of DataFrames
    
    # Duration uses operations rather than time. Ex. at 120Hz, 120 = 1 sec
    for trial in range(1, int(trials) + 1):
        # d = int(time * HZ)#d = np.random.randint(2 * HZ, (8 * HZ) + 1) # Min = 2s, Max = 8s
        d = int(duration_values[trial - 1] * HZ)
        traj[trial] = run_trial(d, a, l)
    
    # Show full DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    #a = a
    #l = l
    
    display_results(1)
    if dpg.does_item_exist("Trajectory Plot"):
        dpg.delete_item("Trajectory Plot")
        
    with dpg.window(label="Trajectory Map", width=526, height=515, tag="Trajectory Plot", no_title_bar=True, no_move=True):
        with dpg.tab_bar(tag='tab_bar'):
            dpg.add_tab(label='Trajectory Plot')
            dpg.add_tab(label='Laser')
            dpg.add_tab(label='All Combinations')
        dpg.set_item_pos("Trajectory Plot", (475, 0))

    show_plot(1)
    show_plots()
    
    return a, l, duration_values

def pursuit_sim_all():
    global traj_all_combinations, trials

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
            for trial in range(1, int(trials) + 1):  # Assuming 50 trials
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

    with dpg.window(label="Plot Distributions", width=w_size, height=h_size, tag="Plot Distributions", no_title_bar=True, no_move=True):
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

# Function to regenerate trajectories per index
def regen_trajectory(index):
    global traj, a, l, time, HZ, df
    print(f'Regenerating Trial {index}...')
    d = int(time * HZ)
    traj[index] = run_trial(d, a, l)
    show_plot(index)

# Function to flip y per index
def flip_y(index):
    global traj
    #print(f'Index = {index}')
    traj[index]['Y'] = -traj[index]['Y']
    traj[index]['Norm Y'] = -traj[index]['Norm Y']
    show_plot(index)

# Function to flip x per index
def flip_x(index):
    global traj
    #print(f'Index = {index}')
    traj[index]['X'] = -traj[index]['X']
    traj[index]['Norm X'] = -traj[index]['Norm X']
    show_plot(index)

# Function to extend trajectories per index
def extend_trajectory(index):
    global traj, HZ, adjusted_time
    #print(f'Index = {index}')
    traj[index] = pd.concat([traj[index], traj[index].iloc[int(HZ * adjusted_time):][::-1]]).reset_index(drop=True)
    show_plot(index)

# Function to trim trajectories per index
def trim_trajectory(index):
    global traj, HZ, adjusted_time
    #print(f'Index = {index}')
    trim_index = len(traj[index]) - int(HZ * adjusted_time)
    traj[index] = traj[index].iloc[:int(trim_index)].reset_index(drop=True)
    show_plot(index)

# Function to scale trajectories per index
def scale_trajectory(index):
    global traj, HZ, adjusted_time
    df['Norm X'] = (2 * scale_x) * ((df['X'] - min_x) / (max_x - min_x)) - scale_x
    df['Norm Y'] = (2 * scale_y) * ((df['Y'] - min_y) / (max_y - min_y)) - scale_y
    show_plot(index)

# Run the GUI
create_gui()

