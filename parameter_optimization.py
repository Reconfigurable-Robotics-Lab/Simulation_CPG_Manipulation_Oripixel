#
# Copyright (c) 2025 Yuhao Jiang, RRL, EPFL
# license: Apache-2.0 license
#


import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import json
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
from Oripixel import Oripixel_manipulation


offset_inputs_bounds = [0, np.pi]
deltas = [60, 150]
object_size = [[0.15, 0.15, 5e-3]]
object_mass = [0.264]

def inv_kinematic(delta, psi, r0, r, l, lower_limits, upper_limits, legconfig):
    
    eps = 1e-8 
    delta = delta * np.pi / 180
    psi = psi * np.pi / 180
    psi = psi / 2
    r0 = r0 / np.sin(np.pi / 2 - psi)

    sols = np.zeros(3)
    phi = np.zeros(2) 

    for i in range(3):
        a = (r - l) * (np.sin(psi) * np.cos(delta - legconfig[i])) - (r0 / 2)
        b = 2 * l * np.cos(psi)
        c = (r + l) * (np.sin(psi) * np.cos(delta - legconfig[i])) - (r0 / 2)

        root = b * b - 4 * a * c
        
        if root >= 0: 

            phi[0] = (-b + np.sqrt(root)) / (2 * (a + eps))
            phi[1] = (-b - np.sqrt(root)) / (2 * (a + eps))

            phi[0] = 2 * np.arctan2(phi[0], 1)
            phi[1] = 2 * np.arctan2(phi[1], 1)

            valid = False
            for j in range(2):
                if lower_limits[i] < phi[j] < upper_limits[i]:
                    sols[i] = np.rad2deg(phi[j])  # Convert to degrees
                    valid = True
                    break 

            if not valid:
                return None
        else:

            return None

    return sols

def within_workspace(rec, delta):

    t_total = 4  
    sim_step = 5e-4 
    steps = int(t_total / sim_step) 
    no_ori_pixels = 5  
    h_amp, freq, h_offset, psi_amp, psi_offset, offset_inputs, phase_shift1, threshold = rec

    t_sim = 0
    settling_time = 1
    for _ in range(steps):
        if t_sim >= settling_time:
            t = t_sim - settling_time

            for m in range(no_ori_pixels):
                if m % 2 == 0:
                    phase_shift = 0
                else:
                    phase_shift = phase_shift1

                height = h_amp * np.sin(2 * np.pi * freq * t + phase_shift) + h_offset
                psi = psi_amp * np.sin(2 * np.pi * freq * t + offset_inputs + phase_shift) + psi_offset

                motor_ang = inv_kinematic(delta, psi, height, 20.21e-3, 30e-3, np.deg2rad([1, 1, 1]), np.deg2rad([89,89,89]), [0, 2*np.pi/3, 4*np.pi/3])

                if motor_ang is None or np.any(motor_ang) is None:
                    return False
        
        else:
            t = t_sim

        t_sim += sim_step

    return True

def append_to_json_file(filename, data):
    with open(filename, 'r+') as f:
        file_data = json.load(f)
        file_data.append(data)
        f.seek(0)
        json.dump(file_data, f, indent=4)

def obj(x: pd.DataFrame, size, mass, delta) -> np.ndarray:


    x_values = x[['h_amp', 'freq', 'h_offset', 'psi_amp', 'psi_offset', 'offset_inputs', 'phase_shift1', 'threshold']].values

    num_x = x_values.shape[0] 

    ret = np.zeros((num_x, 1))

    valid_suggestions = []
    for i in range(num_x):

        params = np.array([
                x_values[i, 0],  # h_amp
                x_values[i, 1],  # freq
                x_values[i, 2],  # h_offset
                x_values[i, 3],  # psi_amp
                x_values[i, 4],  # psi_offset
                x_values[i, 5],  # offset_inputs
                x_values[i, 6],   # phase_shift1
                x_values[i, 7]   # threshold
            ])

        if not within_workspace(params, delta):
            # Apply a large positive penalty for out-of-bound parameters
            ret[i, 0] = 1000000  
        else:
            # Add valid suggestion to the list as a tuple (i, params)
            valid_suggestions.append((i, params))

    if valid_suggestions:
        rewards = sim_all(valid_suggestions, size, mass, delta) 

        for suggestion_id, result in rewards:

            ret[int(suggestion_id)] = result

    return ret

def scale_servo_poses(x, x_min, x_max, y_min, y_max):
    return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min

def mycontroller_CPG(t, params, delta):
    global prev_mot_angs, heights, psis
    
    time_current = t
    no_rows = 5 # Number of rows
    no_ori_pixels = 5 # Number of ori-pixels in a row
    servo_poses = np.zeros(3*no_rows*no_ori_pixels)
    
    for i in range(no_ori_pixels):
        for m in range(no_rows):
            h_amp = params[0]
            freq = params[1]
            h_offset = params[2]
            psi_amp = params[3]
            psi_offset = params[4]
            offset_inputs = params[5]
            if m % 2 == 0: # Alternating groups from row to row
                if i % 2 == 0:
                    phase_shift = 0
                else:
                    phase_shift = params[6]
            else:
                if i % 2 == 0:
                    phase_shift = params[6]
                else:
                    phase_shift = 0 
            
            height = h_amp * np.sin(2*np.pi*freq * time_current + phase_shift) + h_offset
            psi = psi_amp * np.sin(2*np.pi*freq * time_current + offset_inputs + phase_shift) + psi_offset  # in degrees

            
            
            motor_ang = inv_kinematic(delta, psi, height, 20.21e-3, 30e-3, np.deg2rad([1, 1, 1]), np.deg2rad([89,89,89]), [0, 2*np.pi/3, 4*np.pi/3])
            motor_ang = scale_servo_poses(motor_ang, 0, 90, 6, 60)
            
            servo_poses[no_rows*3*i + 3*m] = np.radians(motor_ang[0])
            servo_poses[no_rows*3*i + 3*m + 1] = np.radians(motor_ang[1])
            servo_poses[no_rows*3*i + 3*m + 2] = np.radians(motor_ang[2])

            prev_mot_angs[i,:,m] = motor_ang[:]
    
    return servo_poses

def run_mujoco_selected_tiles(params, threshold = None, render_mode=None, load=False, verbose=False):
    global prev_mot_angs
    params, object_size, object_mass, delta = params

    suggestion_id = params[0]
    params = np.array(params[1])
    
    states = []
    
    prev_mot_angs = np.zeros((5,3,5))
    t_total = 15
    if load:
        t_total = 15
    sim_step = 5e-4

    if delta == 150:
        if params[5] >= np.pi:
            object_pos = [120e-3,360e-3]
        else:
            object_pos = [360e-3, 360e-3]
    else:
        if params[5] >= np.pi:
            object_pos = [120e-3, 120e-3]
        else:
            object_pos = [120e-3, 360e-3]


    robot = Oripixel_manipulation(x_size=5, y_size=5, x_group_size=1, y_group_size=1, 
                                  kp=1, sim_step = sim_step, render_mode=render_mode,
                                  mass_object=object_mass, size_object=object_size, 
                                  object_x=[0], object_y=[0], x_object_offset=object_pos[0], y_object_offset = object_pos[1])
    s = robot.reset()
    servo_pos_init = [np.radians(45)] * 75
    t_sim = 0
    steps = int(t_total/sim_step)
    servo_pos = 0
    settling_time = 1
    data_sampling_rate = 1000
    next_data_sample_time = 0

    if threshold is None:
        threshold = params[-1]
    
    for step in range(steps):
        t_sim = step * sim_step
        if t_sim >= settling_time:
            t = t_sim - settling_time
            tiles_covered = robot.which_tile(threshold)

            servo_pos = mycontroller_CPG(t, params, delta)
            for i in range(25):
                if i not in tiles_covered[0]:
                    servo_pos[3*i:3*i+3] = [np.radians(85)]*3
            s = robot.step(servo_pos)
            if t_sim >= next_data_sample_time:
                states = np.vstack((states, s))
                next_data_sample_time += 1/data_sampling_rate
        else:
            s = robot.step(servo_pos_init)
            initial_pos = s[1:4]
            initial_rot = s[4:7]
            states = s
        
        if render_mode is not None:
            frame = robot.render()
        
    rot = states[:,4:7]
    final_pos = states[-1,1:4]
    final_rot = states[-1,4:7]
    roll_vel = []
    pitch_vel = []
    yaw_vel = []
    i = 0

    while i+5 < rot.shape[0]:
        velocity = np.abs((rot[i+5,:]-rot[i,:])/(5*sim_step))
        i = i+5
        roll_vel.append(velocity[0])
        pitch_vel.append(velocity[1])
        yaw_vel.append(velocity[2])
    

    roll_avg = sum(roll_vel)/len(roll_vel)
    pitch_avg = sum(pitch_vel)/len(pitch_vel)
    yaw_avg = sum(yaw_vel)/len(yaw_vel)
    rot_penalty = np.deg2rad(roll_avg + pitch_avg + yaw_avg)

    z_penalty = np.std(states[:,3])

    if delta == 60:
        distance = np.abs(final_pos[1] - initial_pos[1])
    elif delta == 150:
        distance = np.abs(final_pos[0] - initial_pos[0])
    
    cost = -distance*0.5 + rot_penalty + z_penalty*5

    robot.reset()

    
    velocity = (distance/object_size[0])/t_total

    # Data collection
    data = {
        "x_displacement": states[:, 1],
        "y_displacement": states[:, 2],
        "z_displacement": states[:, 3],
        "x_vel": (np.abs(final_pos[0] - initial_pos[0])/object_size[0])/t_total,
        "y_vel": (np.abs(final_pos[1] - initial_pos[1])/object_size[0])/t_total,
        "z_vel": (np.abs(final_pos[2] - initial_pos[1])/object_size[0])/t_total,
        "roll": rot[:, 0],
        "pitch": rot[:, 1],
        "yaw": rot[:, 2],
    }
    print(f"x vel: {data['x_vel']:.4f} body lengths/sec, y vel: {data['y_vel']:.4f} body lengths/sec")

    if verbose:
        print(f"The object of size: {object_size} and mass: {object_mass} traveled a distance of {distance:.5f} m in {t_total} seconds.")
        print(f"The velocity of the object is {velocity:.2f} object lengths per second")
    
    if load:
        return data
    else:
        return [suggestion_id, cost]

def sim_all(valid_suggestions, size, mass, delta):

    with Pool(processes=len(valid_suggestions)) as pool:
        rewards = pool.map(run_mujoco_selected_tiles, [(params, size, mass, delta) for params in valid_suggestions])

    return rewards


if __name__ == "__main__":   
    for delta in deltas:
        for off_inp_bound in offset_inputs_bounds:
            timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")

            filename = f'optimization_results_{timestamp}_bound_{str(off_inp_bound)}_delta{delta}.json'

            with open(filename, 'w') as f:
                json.dump([], f)
            
            for size in object_size:
                for mass in object_mass:
                    reward_history = []
                    print(f"Running HEBO optimization for object size of: {size} and object mass of: {mass} kg")
                    # Create design space
                    space = DesignSpace().parse([
                        {'name': 'h_amp', 'type': 'num', 'lb': 0.005, 'ub': 0.04},
                        {'name': 'freq', 'type': 'num', 'lb': 0.5, 'ub': 0.8},
                        {'name': 'h_offset', 'type': 'num', 'lb': 0.02, 'ub': 0.04},
                        {'name': 'psi_amp', 'type': 'num', 'lb': 20, 'ub': 45},
                        {'name': 'psi_offset', 'type': 'num', 'lb': -15, 'ub': 15},
                        {'name': 'offset_inputs', 'type':'num', 'lb': 0+off_inp_bound, 'ub': np.pi+off_inp_bound},
                        {'name': 'phase_shift1', 'type': 'num', 'lb':0,'ub': 2*np.pi},
                        {'name': 'threshold', 'type': 'num', 'lb': 0.0, 'ub': 0.5},
                    ])

                    # Run HEBO optimization
                    opt = HEBO(space, rand_sample=50)
                    for i in range(500):
                        suggestion_number = 100
                        rec = opt.suggest(n_suggestions=suggestion_number)
                        
                        # Run Mujoco simulation
                        opt.observe(rec, obj(rec, size, mass, delta))

                        print(
                            f'iter: {i+1}, '
                            f'min: {opt.y.min():.3f}, '
                            f'x: {list(opt.X.values[np.argmin(opt.y)])}'
                        )
                        reward_history.append(opt.y.min())
                        
                        # Early stopping condition
                        threshold = 25
                        if len(reward_history) >= threshold:
                            recent_rewards = reward_history[-threshold:]
                            difference = recent_rewards[-1]-recent_rewards[0]
                            if abs(difference) < 0.005:
                                print("Stopping early due to minimal change in reward.")
                                break

                    best_params = list(opt.X.values[np.argmin(opt.y)])
                    data_to_save = {
                        'size': size,
                        'mass': mass,
                        'best_reward': opt.y.min(),
                        'best_params': best_params,
                        'reward_history': reward_history
                    }

                    append_to_json_file(filename, data_to_save)

                    print(f"Results for size {size} and mass {mass} appended to {filename}")
