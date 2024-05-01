import csv

from scapy.all import *
from scapy.layers.inet import TCP
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import os
import gen_main_csv_trans
import random

normal_pcap_file_folder = ""
dos_pcap_file_folder = ""
injection_pcap_file_folder = ""
scan_pcap_file_folder = ""
temp_session_dir = ''
sharkpath = ""
group_size = 256

main_csv_path = ""
main_csv_path_test = ""
all_max = 54
normal_max = 240
dos_max = 54
inj_max = 54
scan_max = 54
train_test_ratio = 1.0

print("all: " + str(all_max * 2 + normal_max + dos_max))
print("proportion: "+ str((all_max * 2 + dos_max) / (all_max * 2 + normal_max + dos_max)))
def remove_session_pcaps(temp_session_dir):
    for filename in os.listdir(temp_session_dir):
        file_path = os.path.join(temp_session_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def split_all_pcaps(folder_path, temp_session_dir):
    if not os.path.exists(folder_path):
        print("Folder not found")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".pcap"):
            file_path = os.path.join(folder_path, filename)
            print("Splitting" + file_path)
            command = "\"" + sharkpath + "/editcap.exe " + "\"" + " -F pcap " + str(file_path) + \
                      " - | SplitCap.exe -r - -s session -o " + str(temp_session_dir)
            os.system(command)


def read_normal_states(temp_session_dir):
    states = {}
    counter = 0
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            packets = rdpcap(file_path)
            tcp_packets = [packet for packet in packets if TCP in packet]
            for packet in tcp_packets:
                tcp_flag = packet[TCP].flags
                if tcp_flag not in states.values():
                    states[counter] = tcp_flag
                    counter += 1
    return states


def read_dos_states(temp_session_dir):
    states = {}
    counter = 0
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            packets = rdpcap(file_path)
            tcp_packets = [packet for packet in packets if TCP in packet]
            for packet in tcp_packets:
                tcp_flag = packet[TCP].flags
                if tcp_flag not in states.values():
                    states[counter] = tcp_flag
                    counter += 1
    return states


def read_injection_states(temp_session_dir):
    states = {}
    counter = 0
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            packets = rdpcap(file_path)
            tcp_packets = [packet for packet in packets if TCP in packet]
            for packet in tcp_packets:
                tcp_flag = packet[TCP].flags
                if tcp_flag not in states.values():
                    states[counter] = tcp_flag
                    counter += 1
    return states

def read_scan_states(temp_session_dir):
    states = {}
    counter = 0
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            packets = rdpcap(file_path)
            tcp_packets = [packet for packet in packets if TCP in packet]
            for packet in tcp_packets:
                tcp_flag = packet[TCP].flags
                if tcp_flag not in states.values():
                    states[counter] = tcp_flag
                    counter += 1
    return states


def generate_trans_matrix(file_path,stat_dict,matrix):
    packets = rdpcap(file_path)
    last_state = None
    tcp_packets = [packet for packet in packets if TCP in packet]
    for packet in tcp_packets:
        tcp_flag = packet[TCP].flags
        if last_state:
            find_key_last = -1
            for key, value in stat_dict.items():
                if value == last_state:
                    find_key_last = key
            if find_key_last == -1:
                print("Fatal error: Unknown flag")
                exit(1)

            find_key_this = -1
            for key, value in stat_dict.items():
                if value == tcp_flag:
                    find_key_this = key
            if find_key_this == -1:
                print("Fatal error: Unknown flag")
                exit(1)
            matrix[find_key_last][find_key_this] = 1
        last_state = tcp_flag
    return matrix


def gen_csv(id, packet,  matrix, graph_id, states_dic, packets, filename, session):
    next_states = []
    features = {}
    if not packet.haslayer(TCP):
        return -1
    tcp_flag = packet[TCP].flags
    find_key_this = -1
    for key, value in states_dic.items():
        if value == tcp_flag:
            find_key_this = key
    if find_key_this == -1:
        print("Fatal error: Unknown flag")
        exit(1)
    next_keys = matrix[find_key_this]

    for index,key_2 in enumerate(next_keys):
        if key_2 == 1:
            next_states.append(int(states_dic[index]))

    next_states = str(next_states)
    if packet.haslayer(TCP):
        tcp_header = packet[TCP]
        features = {
            "Source Port": tcp_header.sport,
            "Destination Port": tcp_header.dport,
            "Sequence Number": tcp_header.seq,
            "Acknowledgment Number": tcp_header.ack,
            "Data Offset": tcp_header.dataofs,
            "Reserved": tcp_header.reserved,
            "Flags": int(tcp_header.flags),
            "Window Size": tcp_header.window,
            "Checksum": tcp_header.chksum,
            "Urgent Pointer": tcp_header.urgptr,
        }
        data_row = [id] + [features[key] for key in header[1:-3]] + [next_states, session, graph_id]

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)  # 写入数据行
        return 1


if __name__ == '__main__':

    gen_normal = 1
    gen_dos = 1
    gen_injection = 1
    gen_scan = 1
    gen_main_csv = 1
    session_counter = 1
    counter_package = 0
    session_counter_test = 1
    counter_package_test = 0
    header = ["ID", "Source Port", "Destination Port", "Sequence Number", "Acknowledgment Number",
                "Data Offset", "Reserved", "Flags", "Window Size", "Checksum", "Urgent Pointer",
                "next_packages", "Session", "Label"]
    if os.path.isfile(main_csv_path) or os.path.islink(main_csv_path):
        os.unlink(main_csv_path)
    if os.path.isfile(main_csv_path_test) or os.path.islink(main_csv_path_test):
        os.unlink(main_csv_path_test)
    with open(main_csv_path, 'w') as csv_file:
        csv_file.write(','.join(header) + '\n')
    if gen_normal:
        remove_session_pcaps(temp_session_dir)
        split_all_pcaps(normal_pcap_file_folder, temp_session_dir)
        for filename in os.listdir(temp_session_dir):
            if "UDP" in filename:
                file_path = os.path.join(temp_session_dir, filename)
                os.unlink(file_path)
        states_dic = read_normal_states(temp_session_dir)
        size = len(states_dic)
        normal_counter = 0
        normal_static_matrix = [[0 for _ in range(size)] for _ in range(size)]

        for filename in os.listdir(temp_session_dir):
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                normal_static_matrix = generate_trans_matrix(file_path, states_dic, normal_static_matrix)

        print("All normal states:")
        print(states_dic)
        for column in normal_static_matrix:
            print(column)

        normal_session_train = []
        all_session_count = len(os.listdir(temp_session_dir))
        print(all_session_count)
        selected = random.sample(range(0, all_session_count), normal_max)
        print(selected)
        for filename in os.listdir(temp_session_dir):
            if normal_counter in selected:
                normal_session_train.append(filename)
                print("insert")
            normal_counter += 1
            print(normal_counter)
        print(normal_session_train)
        for filename in normal_session_train:
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                packets = rdpcap(file_path)
                session_plus = False
                for single_packet in packets:

                    output = gen_csv(counter_package, single_packet, normal_static_matrix, "normal", states_dic
                            , packets, main_csv_path, session_counter)
                    if output > 0:
                        counter_package += 1
                        session_plus = True
                if session_plus:
                    session_counter += 1

    if gen_dos:
        print(counter_package)

        remove_session_pcaps(temp_session_dir)
        split_all_pcaps(dos_pcap_file_folder, temp_session_dir)
        for filename in os.listdir(temp_session_dir):
            if "UDP" in filename:
                file_path = os.path.join(temp_session_dir, filename)
                os.unlink(file_path)
        states_dic = read_dos_states(temp_session_dir)
        size = len(states_dic)

        dos_static_matrix = [[0 for _ in range(size)] for _ in range(size)]
        dos_counter = 0
        dos_session_train = []
        all_session_count = len(os.listdir(temp_session_dir))
        selected = random.sample(range(0, all_session_count), dos_max)
        print(selected)
        for filename in os.listdir(temp_session_dir):
            if dos_counter in selected:
                dos_session_train.append(filename)
            dos_counter += 1

        for filename in os.listdir(temp_session_dir):
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                dos_static_matrix = generate_trans_matrix(file_path, states_dic, dos_static_matrix)

        print("All DOS states:")
        print(states_dic)
        for column in dos_static_matrix:
            print(column)

        for filename in dos_session_train:
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                packets = rdpcap(file_path)
                session_plus = False
                for single_packet in packets:

                    output = gen_csv(counter_package, single_packet, dos_static_matrix, "dos", states_dic
                            , packets, main_csv_path, session_counter)
                    if output > 0:
                        counter_package += 1
                        session_plus = True
                if session_plus:
                    session_counter += 1

    if gen_injection:
        remove_session_pcaps(temp_session_dir)
        split_all_pcaps(injection_pcap_file_folder, temp_session_dir)
        states_dic = read_injection_states(temp_session_dir)
        size = len(states_dic)
        injection_static_matrix = [[0 for _ in range(size)] for _ in range(size)]
        for filename in os.listdir(temp_session_dir):
            if "UDP" in filename:
                file_path = os.path.join(temp_session_dir, filename)
                os.unlink(file_path)
        for filename in os.listdir(temp_session_dir):
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                injection_static_matrix = generate_trans_matrix(file_path, states_dic, injection_static_matrix)

        print("All INJECTION states:")
        print(states_dic)
        for column in injection_static_matrix:
            print(column)

        injection_counter = 0
        injection_session_train = []
        injection_session_test = []
        all_session_count = len(os.listdir(temp_session_dir))
        selected = random.sample(range(0, all_session_count), inj_max)
        print(selected)
        for filename in os.listdir(temp_session_dir):
            if injection_counter in selected:
                injection_session_train.append(filename)
            injection_counter += 1

        for filename in injection_session_train:
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                packets = rdpcap(file_path)
                session_plus = False
                for single_packet in packets:

                    output = gen_csv(counter_package, single_packet, injection_static_matrix, "injection", states_dic
                            , packets, main_csv_path, session_counter)
                    if output > 0:
                        counter_package += 1
                        session_plus = True
                if session_plus:
                    session_counter += 1

    if gen_scan:
        remove_session_pcaps(temp_session_dir)
        split_all_pcaps(scan_pcap_file_folder, temp_session_dir)
        states_dic = read_scan_states(temp_session_dir)
        size = len(states_dic)
        scan_static_matrix = [[0 for _ in range(size)] for _ in range(size)]
        for filename in os.listdir(temp_session_dir):
            if "UDP" in filename:
                file_path = os.path.join(temp_session_dir, filename)
                os.unlink(file_path)
        for filename in os.listdir(temp_session_dir):
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                scan_static_matrix = generate_trans_matrix(file_path, states_dic, scan_static_matrix)



        print("All INJECTION states:")
        print(states_dic)
        for column in scan_static_matrix:
            print(column)


        scan_counter = 0
        scan_session_train = []
        all_session_count = len(os.listdir(temp_session_dir))
        selected = random.sample(range(0, all_session_count), scan_max)
        print(selected)
        for filename in os.listdir(temp_session_dir):
            if scan_counter in selected:
                scan_session_train.append(filename)
            scan_counter += 1

        for filename in scan_session_train:
            if filename.endswith(".pcap"):
                file_path = os.path.join(temp_session_dir, filename)
                packets = rdpcap(file_path)
                session_plus = False
                for single_packet in packets:

                    output = gen_csv(counter_package, single_packet, scan_static_matrix, "scan", states_dic
                            , packets, main_csv_path, session_counter)
                    if output > 0:
                        counter_package += 1
                        session_plus = True
                if session_plus:
                    session_counter += 1
    if gen_main_csv:
        gen_main_csv_trans.gen_main_csv()
