import csv
import itertools
import sys
import re
import math

def get_root_mean_square( mean_square, number):
    return math.sqrt(mean_square / number)

def gpsr_tlm_compare(target_arr, answer_arr, lift_off_time, fileobj, csv_header):
    cache_idx = 0
    sim_data_list = []
    start_flight_idx = 0
    iter_idx = 0
    mean_len_sq = 0.0
    mean_speed_sq = 0.0
    filecusor = csv.writer(fileobj)
    filecusor.writerow(csv_header)
    for target_elem in enumerate(target_arr):
        sim_data_list = []
        iter_idx += 1
        if target_elem[0] == 0:
            continue;
        if float(target_elem[1][0]) == lift_off_time:
            start_flight_idx = iter_idx
        for answer_elem in enumerate(answer_arr , start = cache_idx):
            cache_idx = answer_elem[0]
            if answer_elem[0] == 0:
                continue;
            if abs(float(target_elem[1][0]) - float(answer_elem[1][0])) == 0.0:
                # simtime
                sim_data_list.append(target_elem[1][0])
                # gps sow time
                sim_data_list.append(target_elem[1][1])
                # DM Length
                dm_length = math.sqrt(float(answer_elem[1][2])**2 + float(answer_elem[1][3])**2  + float(answer_elem[1][4])**2)
                sim_data_list.append(dm_length)
                # DM SPEED
                dm_speed = math.sqrt(float(answer_elem[1][5])**2 + float(answer_elem[1][6])**2  + float(answer_elem[1][7])**2)
                sim_data_list.append(dm_speed)
                # DM ABEE
                dm_abee = float(answer_elem[1][10])
                sim_data_list.append(dm_abee)
                # Target Benchmark (DM_GPSR_TLM - target_GPSR_TLM)
                target_length_err = float(answer_elem[1][18]) - float(target_elem[1][18])
                target_speed_err = float(answer_elem[1][19]) - float(target_elem[1][19])
                sim_data_list.append(target_length_err)
                sim_data_list.append(target_speed_err)
                # Answer DM-TLM
                sim_data_list.append(answer_elem[1][20])
                sim_data_list.append(answer_elem[1][21])
                # Target DM-TLM
                sim_data_list.append(target_elem[1][20])
                sim_data_list.append(target_elem[1][21])

                filecusor.writerow(sim_data_list)
                # Root Mean square
                if iter_idx >= start_flight_idx:
                    mean_len_sq = mean_len_sq + target_length_err**2
                    mean_speed_sq = mean_speed_sq + target_speed_err**2
                break
    return (iter_idx - start_flight_idx), mean_len_sq, mean_speed_sq
