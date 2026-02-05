#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:02:25 2025

@author: idiot
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from pyomo.environ import *
import pyomo.environ as pyo
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import griddata
from collections import defaultdict
import math
from pyomo.opt import TerminationCondition

#%%
    
def optimization_process(dict_test,
                         veh_length = 4.5,
                         speed_limit = 22.352, 
                         x_weight = 0.00005,
                         a_weight = 0.1,
                         error_weight = 10,
                         acc_low = -3.5, # acceleration lower threshold
                         acc_high = 3.5, # acceleration upper threshold
                         M=10000, # big M method
                         attack_time_gap = 15,
                         extend_time_gap = 5,
                         # --- FIX IS HERE ---
                         initial_attack_time = 0, # the time when the attack starts, default 0
                         # --- A red_phase flag ---
                         red_phase = False
                         ):

    #auxiliary functions
    def Preparation_for_optimization_indexed(dict_test_input, attack_time_gap, extend_time_gap, interval = 1):
        
        '''
        Data Preparation for Optimization
            attack_time_gap: time duration between generate and implement attack
            extend_time_gap: time extension for attack
        '''
        
        dict_test = dict_test_input.copy()   
        veh_id_list = list(dict_test.keys())
        
        N = len(dict_test) #number of vehicles
        T = attack_time_gap+extend_time_gap+1
      
        start_dist={}
        start_v={}
        arrival_time={}
        detected_time={}
        detected_dist={}
        detected_v={}
        departure_time={}
        lane={}
        for veh_id in veh_id_list:
            start_dist[veh_id_list.index(veh_id)]=dict_test[veh_id][0]
            start_v[veh_id_list.index(veh_id)]=dict_test[veh_id][1]
            detected_dist[veh_id_list.index(veh_id)]=dict_test[veh_id][2]
            detected_v[veh_id_list.index(veh_id)]=dict_test[veh_id][3]
            lane[veh_id_list.index(veh_id)]=dict_test[veh_id][4]
            arrival_time[veh_id_list.index(veh_id)]=0
            detected_time[veh_id_list.index(veh_id)]=0+attack_time_gap
            departure_time[veh_id_list.index(veh_id)]=0+attack_time_gap+extend_time_gap
     
        time_sequence=list(range(T))
        return N, T, start_dist, start_v, arrival_time, detected_dist, detected_v, detected_time, departure_time, lane, time_sequence
    
    def replay_results_complete(dict_test_input, merged_df, initial_attack_time):
        
        opt_result = merged_df.copy()
        
        all_veh = dict_test_input.copy()
        veh_id_list = list(all_veh.keys())
        
        lane_list = list(set(merged_df.Lane))
        
        for i in range(len(opt_result)):
            veh_id_temp = opt_result.Vehicle.iloc[i]
            time_id_temp = int(opt_result.Time.iloc[i])
            
            # --- FIX: Use .loc for reliable assignment ---
            opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
            opt_result.loc[opt_result.index[i], 'Time'] = time_id_temp
            # --- END FIX ---

        opt_result=opt_result.dropna(subset=['v'])

        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        
        # for j in range(len(lane_list)):
        #     l = lane_list[j]
        #     data_veh_l = opt_result[opt_result.Lane == l]
        #     plt.figure(dpi=200)
            
        #     first = True
            
        #     for veh_id in veh_id_list:
        #         if first:
        #             data_veh_temp = data_veh_l[data_veh_l.Vehicle == veh_id]
        #             # Check if data exists for this vehicle in this lane before plotting to avoid errors
        #             if not data_veh_temp.empty:
        #                 plt.scatter(data_veh_temp['Time'], data_veh_temp['x'], color='green', s=8, label='Fake Trajectory')
        #                 plt.plot(data_veh_temp['Time'], data_veh_temp['x'], color='green',linewidth=2)
        #                 first = False
        #         else:
        #             data_veh_temp = data_veh_l[data_veh_l.Vehicle == veh_id]
        #             if not data_veh_temp.empty:
        #                 plt.scatter(data_veh_temp['Time'], data_veh_temp['x'], color='green', s=8)
        #                 plt.plot(data_veh_temp['Time'], data_veh_temp['x'], color='green',linewidth=2)

        #     plt.title(f'Lane: {l}')

        #     plt.xlim(min(opt_result.Time), max(opt_result.Time))
        #     plt.ylim(max(opt_result.x), min(opt_result.x)) ###01192025 hardcode
        #     plt.xlabel('Time (seconds)')
        #     plt.ylabel('Distance (meters)')
        #     plt.legend(loc='lower right')
        #     plt.grid(True)
            
        #     # Save figure locally with a unique name per lane
        #     # Replaced plt.show() with plt.savefig()
        #     filename = f"lane_{l}_trajectory_{timestamp}.png"
        #     full_path = os.path.join(save_dir, filename)
        #     plt.savefig(full_path)
        #     plt.close() # Close figure to free memory

        # --- UPDATED SECTION STARTS HERE ---
        opt_dict = {}
        
        # Group by Time and Lane (Pandas groupby is optimized)
        for (time_val, lane), group in opt_result.groupby(["Time", "Lane"]):
            time_key = int(time_val + initial_attack_time)
            
            if time_key not in opt_dict:
                opt_dict[time_key] = {}
            
            # VECTORIZED ID GENERATION
            ids = group['Vehicle'].astype(str) + '_' + str(time_key)
            
            # Create dictionary with filtering
            lane_dict = {
                uid: {"speed": v, "lane_pos": x}
                for uid, v, x in zip(ids, group['v'], group['x'])
                if x >= 0  # <--- FILTER: Only keep vehicles behind the stop bar
            }
            
            # Only add the lane to the dictionary if it has vehicles left after filtering
            if lane_dict:
                opt_dict[time_key][lane] = lane_dict
                    
        return opt_dict
    
    
    #Optimization
    N, T, start_dist, start_v, arrival_time, detected_dist, detected_v, detected_time, departure_time, lane, time_sequence = Preparation_for_optimization_indexed(dict_test, attack_time_gap, extend_time_gap)

    # Create a Pyomo model
    model = ConcreteModel()

    model.V = RangeSet(0, N-1)  # Vehicles
    model.T = RangeSet(0, T-1)  # Time steps
    
    #absolute value of a
    model.a_abs = Var(model.V, model.T, domain=NonNegativeReals)
    # Velocity of vehicle v at time t
    model.v = Var(model.V, model.T, domain=NonNegativeReals)
    # Position of vehicle v at time t
    model.x = Var(model.V, model.T, domain=Reals)
    # Auxilliary variable to minimize v at the detected timestamp
    model.u = Var(model.V, model.T, domain=NonNegativeReals)

    # Objective
    if red_phase:
        def objective_rule(model):
            return (
                a_weight * sum(
                    model.v[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t == departure_time[v]
                )
                + a_weight * sum(
                    model.a_abs[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t >= arrival_time[v] and t <= departure_time[v]
                )
                - x_weight * sum(
                    model.x[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t >= arrival_time[v] and t <= departure_time[v]
                )
            )
    else:
        def objective_rule(model):
            return (
                error_weight * sum(
                    model.u[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t == detected_time[v]
                )
                + a_weight * sum(
                    model.a_abs[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t >= arrival_time[v] and t <= departure_time[v]
                )
                - x_weight * sum(
                    model.x[v, t]
                    for t in time_sequence
                    for v in model.V 
                    if t >= arrival_time[v] and t <= departure_time[v]
                )
            )
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    def abs_constraints_u1(model, v, t):
        if t == detected_time[v]:
            return model.u[v, t] >= model.v[v, t] - detected_v[v]
        else:
            return Constraint.Skip
    model.abs_constraints_u1 = Constraint(model.V, model.T, rule=abs_constraints_u1)

    def abs_constraints_u2(model, v, t):
        if t == detected_time[v]:
            return model.u[v, t] >= -(model.v[v, t] - detected_v[v])
        else:
            return Constraint.Skip
    model.abs_constraints_u2 = Constraint(model.V, model.T, rule=abs_constraints_u2)
    
    # Distance, Speed, and Acceleration
    def start_dist_rule(model, v):
        return model.x[v, arrival_time[v]] == start_dist[v]
    model.start_dist = Constraint(model.V, rule=start_dist_rule)
    
    def start_v_rule(model, v):
        return model.v[v, arrival_time[v]] == start_v[v]
    model.start_v = Constraint(model.V, rule=start_v_rule)
    
    def detected_distance_rule(model, v, t):
        if t == detected_time[v]:
            return model.x[v, t] == detected_dist[v]
        else:
            return Constraint.Skip
    model.detected_distance = Constraint(model.V, model.T, rule=detected_distance_rule)
    
    def distance_update_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v, t] == model.x[v, t-1] - model.v[v, t-1]
        else:
            return Constraint.Skip
    model.distance_update = Constraint(model.V, model.T, rule=distance_update_rule)
    
    def velocity_limit_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.v[v, t] <= speed_limit
        else:
            return Constraint.Skip
    model.velocity_limit = Constraint(model.V, model.T, rule=velocity_limit_rule)
    
    def acc_limit_rule(model,v,t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return (acc_low, model.v[v, t] - model.v[v, t-1], acc_high)
        else:
            return Constraint.Skip
    model.acc_limit = Constraint(model.V, model.T, rule=acc_limit_rule)
    
    def acc_abs_1_rule(model, v, t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.a_abs[v, t] >= model.v[v, t] - model.v[v, t-1]
        else:
            return Constraint.Skip
    model.acc_abs_1 = Constraint(model.V, model.T, rule=acc_abs_1_rule)
    
    def acc_abs_2_rule(model, v, t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.a_abs[v, t] >= -(model.v[v, t] - model.v[v, t-1])
        else:
            return Constraint.Skip
    model.acc_abs_2 = Constraint(model.V, model.T, rule=acc_abs_2_rule)
    
    def fixed_order_headway_rule(model, v, u, t):
        if v == u:
            return Constraint.Skip

        # Only same-lane interactions
        if lane[v] != lane[u]:
            return Constraint.Skip

        # Only enforce when both are active
        if t < max(arrival_time[v], arrival_time[u]) or t > min(departure_time[v], departure_time[u]):
            return Constraint.Skip

        # start_dist: larger means more upstream (behind) in your convention.
        # If v starts behind u, then v must remain behind u with >= veh_length gap:
        if start_dist[v] > start_dist[u]:
            # v behind u  => x[v,t] >= x[u,t] + veh_length
            return model.x[v, t] >= model.x[u, t] + veh_length
        elif start_dist[u] > start_dist[v]:
            # u behind v  => x[u,t] >= x[v,t] + veh_length
            return model.x[u, t] >= model.x[v, t] + veh_length
        else:
            return Constraint.Skip

    model.fixed_order_headway = Constraint(model.V, model.V, model.T, rule=fixed_order_headway_rule)
    
    # Solve the model
    solver = SolverFactory("gurobi_direct")
    
    start_time = time.time()
    results = solver.solve(model)
    end_time = time.time()

    optimization_time = end_time - start_time
    print(optimization_time)

    if results.solver.termination_condition != TerminationCondition.optimal:
        print("\n" + "="*50)
        print("!!! OPTIMIZATION FAILED !!!")
        print(f"The solver reported a termination condition of: '{results.solver.termination_condition}'")
        print("This means no optimal solution was found, and the variable values are uninitialized.")
        print("="*50 + "\n")
        
        # Return an empty dictionary to stop the program from crashing
        return {}
    
    def extract_variables_to_dataframe(model, variable_name):
        var = getattr(model, variable_name)
        data = []
        for v in model.V:
            for t in model.T:
                try:
                    value = pyo.value(var[v, t])
                    data.append((v, t, value))
                except:
                    data.append((v, t, None))
        df = pd.DataFrame(data, columns=['Vehicle', 'Time', variable_name])
        return df

    def extract_variables_to_dataframe_3(model, variable_name):
        var = getattr(model, variable_name)
        data = []
        for v in model.V:
            for u in model.V:
                for t in model.T:
                    try:
                        value = pyo.value(var[v, u, t])
                        data.append((v, u, t, value))
                    except:
                        data.append((v, u, t, None))
        df = pd.DataFrame(data, columns=['Vehicle', 'LeadVehicle', 'Time', variable_name])
        return df
    
    v_df = extract_variables_to_dataframe(model, 'v')
    x_df = extract_variables_to_dataframe(model, 'x')
    
    dfs = [v_df, x_df]

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['Vehicle', 'Time'])
        
    merged_df["Lane"] = merged_df["Vehicle"].map(lane)
    
    opt_result = replay_results_complete(dict_test,merged_df, initial_attack_time)
    for ts, vehs in list(opt_result.items()):
        for vid in list(vehs.keys()):
            if vehs[vid].get("x", 0) < 0:
                del vehs[vid]
    return opt_result



#%%
'''
Example Usage
file_path = "/home/idiot/Research/DL Attacker/Fake Vehicle Trajectory Generation/2025-09-02-18-32-29.074894_fake_traj_input.p"
with open(file_path, "rb") as f:
    data = pickle.load(f)
dict_test = data[122]
opt_result = optimization_process(dict_test,attack_time_gap = 15, extend_time_gap = 20)
'''