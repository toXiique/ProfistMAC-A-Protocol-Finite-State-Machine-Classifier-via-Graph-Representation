# -*- coding: gbk -*-
from scapy.all import *
import glob
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from scapy.compat import raw

main_csv = ""

attr_csv = ""
edge_csv = ""
graph_csv = ""
label_csv = ""

gen_attr = 1
gen_edge = 1
gen_graph = 1
gen_label = 1


node_id = 0

def gen_main_csv():
    if gen_attr:
        if os.path.isfile(attr_csv) or os.path.islink(attr_csv):
            os.unlink(attr_csv)
        print("Generating nodeattrs.csv...")
        with open(attr_csv, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)

            header = ["ID", "Source Port", "Destination Port", "Sequence Number", "Acknowledgment Number",
                      "Data Offset", "Reserved", "Flags", "Window Size", "Checksum", "Urgent Pointer"]
            writer.writerow(header)
            selected_columns = ["ID", "Source Port", "Destination Port", "Sequence Number", "Acknowledgment Number",
                                "Data Offset", "Reserved", "Flags", "Window Size", "Checksum", "Urgent Pointer"]
            with open(main_csv, mode='r', newline='') as infile:
                reader = csv.DictReader(infile)

                for row in tqdm(reader, desc=f"Writing rows from {main_csv}", leave=False, mininterval=30, miniters=500):

                    selected_row = [row[col] for col in selected_columns if col in row]

                    writer.writerow(selected_row)



    if gen_edge:
        print("Generating edge.csv...")
        session_csv_dir = "raw/session"
        if os.path.isfile(edge_csv) or os.path.islink(edge_csv):
            os.unlink(edge_csv)
        for filename in os.listdir(session_csv_dir):
            file_path = os.path.join(session_csv_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        with open(main_csv, mode='r', newline='') as infile_session:
                reader_session = csv.DictReader(infile_session)
                for row_session in reader_session:
                    selected_columns = ["Session"]
                    selected_row = [row_session[col] for col in selected_columns if col in row_session]
                    session = selected_row[0]
                    file_path_session = session_csv_dir + "/" + session + ".csv"
                    file_exists_before_open = os.path.exists(file_path_session)
                    with open(file_path_session, mode='a', newline='') as outfile:
                        writer = csv.writer(outfile)
                        header = ["ID", "Source Port", "Destination Port", "Sequence Number", "Acknowledgment Number",
                            "Data Offset", "Reserved", "Flags", "Window Size", "Checksum", "Urgent Pointer",
                            "next_packages", "Session", "Label"]
                        if not file_exists_before_open:
                            writer.writerow(header)
                        row_to_write = [row_session[col] for col in header if col in row_session]
                        writer.writerow(row_to_write)
        print("Generating edge.csv...")

        with open(edge_csv, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)

            writer.writerow(['source_node', 'destination_node'])
            with open(main_csv, mode='r', newline='') as infile:

                reader = csv.DictReader(infile)

                for row in tqdm(reader, desc=f"Writing rows from {main_csv}", leave=False, mininterval=30,
                                miniters=500):
                    selected_columns = ["ID", "next_packages", "Session"]
                    selected_row = [row[col] for col in selected_columns if col in row]
                    array = json.loads(selected_row[1])
                    file_to_ergonic = session_csv_dir + "/" + selected_row[2] + ".csv"


                    with open(file_to_ergonic, mode='r', newline='') as infile_2:
                        reader2 = csv.DictReader(infile_2)
                        counter = 0
                        all_counter = 0
                        for row_2 in reader2:
                            all_counter += 1
                            selected_columns_2 = ["ID", "Flags"]

                            selected_row_2 = [row_2[col_2] for col_2 in selected_columns_2 if col_2 in row_2]

                            if int(selected_row_2[1]) in array and selected_row[0] <= selected_row_2[0]:
                                writer.writerow([selected_row[0], selected_row_2[0]])
                                counter += 1
                    print(selected_row[0], counter, all_counter)

    if gen_graph:
        node_id = 0
        if os.path.isfile(graph_csv) or os.path.islink(graph_csv):
            os.unlink(graph_csv)
        print("Generating graph.csv...")

        with open(graph_csv, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)

            writer.writerow(['node_id', 'graph_id'])
            with open(main_csv, mode='r', newline='') as infile_2:
                reader = csv.DictReader(infile_2)
                for row_2 in tqdm(reader, desc=f"Writing rows from {main_csv}", leave=False, mininterval=30,
                                miniters=500):
                    selected_columns_graph = ["ID", "Session"]
                    selected_row_graph = [row_2[col_2] for col_2 in selected_columns_graph if col_2 in row_2]
                    writer.writerow([selected_row_graph[0], selected_row_graph[1]])

    if gen_label:
        label_dict = {"normal": 0, "dos": 1, "injection": 2, "scan": 3}
        Last_session = None
        if os.path.isfile(label_csv) or os.path.islink(label_csv):
            os.unlink(label_csv)
        print("Generating label.csv...")

        with open(main_csv, mode='r', newline='') as infile_session:
                reader_session = csv.DictReader(infile_session)
                for row_session in reader_session:
                    selected_columns = ["Session", "Label"]
                    selected_row = [row_session[col] for col in selected_columns if col in row_session]
                    session = selected_row[0]
                    if not Last_session or Last_session != session:
                        file_exists_before_open = os.path.exists(label_csv)
                        with open(label_csv, mode='a', newline='') as outfile:
                            writer = csv.writer(outfile)
                            if not file_exists_before_open:
                                writer.writerow(['graph_no', 'label'])
                            writer.writerow([selected_row[0], label_dict[selected_row[1]]])
                        Last_session = session

