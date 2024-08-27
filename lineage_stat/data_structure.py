import os
from pickle import load

import pandas as pd
from treelib import Tree
import numpy as np


def get_combined_lineage_tree(time_frame_resolution=1.39,
                              life_span_tree_path=r'Data/LifeSpan'):
    # same as f'{time_int:03}'
    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']
    basic_embryo_name = '200710hmr1plc1p3'
    raw_data_path = r'E:\ProjectData\MembraneProject\AllRawData'
    # embryo_cell_trees_path=sorted(glob.glob(os.path.join(life_span_tree_path, "*")))
    # 0 min in lineage set to ABa ABp last moment, the frame ABal and ABpl appear (AB4) is the 1 frame
    tree_dict = {}
    begin_frame = {}
    for embryo_name in embryo_names:
        cell_tree_file_path = os.path.join(life_span_tree_path, '{}_cell_life_tree'.format(embryo_name))
        with open(cell_tree_file_path, 'rb') as f:
            # print(f)
            tree_dict[embryo_name] = Tree(load(f))
        begin_frame[embryo_name] = max(tree_dict[embryo_name].get_node('ABa').data.get_time()[-1],
                                       tree_dict[embryo_name].get_node('ABp').data.get_time()[-1])
    # print(begin_frame)

    # use sample06 build basic tree
    basic_embryo_cd_file_path = os.path.join(raw_data_path, basic_embryo_name, r"CD{}.csv".format(basic_embryo_name))
    tree_distance_num = 12
    name_dictionary_file_path = os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables',
        r"name_dictionary_no_name.csv")
    max_time = 100
    cell_tree_final = construct_basic_cell_name_tree(cd_file_path=basic_embryo_cd_file_path, max_time=max_time,
                                                     tree_distance_num=tree_distance_num,
                                                     name_dictionary_path=name_dictionary_file_path)

    for node_id in cell_tree_final.expand_tree(sorting=False):
        start_times, end_times = [], []
        for embryo_key in tree_dict.keys():
            # go through all embryo to get average start time and end time
            if not tree_dict[embryo_key].get_node(node_id):
                # print(embryo_key, 'do not exist', node_id)
                pass
            else:
                if len(tree_dict[embryo_key].get_node(node_id).data.get_time()) != 0:
                    # minus the begin frame (AB2 end time)
                    start_times.append(
                        tree_dict[embryo_key].get_node(node_id).data.get_time()[0] - begin_frame[embryo_key])
                    end_times.append(
                        tree_dict[embryo_key].get_node(node_id).data.get_time()[-1] - begin_frame[embryo_key])
        if len(start_times) == 0 or len(end_times) == 0:
            pass
            # print(node_id)
        else:
            cell_tree_final.get_node(node_id).data.set_time([np.average(start_times), np.average(end_times)])
        # print(node_id, 'start hitting cell number', len(start_times), np.average(start_times),
        #       'end hitting cell number', len(end_times), np.average(end_times))

    for node_id in cell_tree_final.expand_tree(sorting=False):
        # start time should use max one, ensure the cell have been appear!
        if node_id != 'P0':
            parent_id = cell_tree_final.parent(node_id).identifier
            if len(cell_tree_final.children(parent_id)) == 2:
                [children1, children2] = cell_tree_final.children(parent_id)
                average_start_time = (children1.data.get_time()[0] + children2.data.get_time()[0]) / 2
                children1.data.get_time()[0] = children2.data.get_time()[0] = average_start_time

            # print(node_id, cell_tree_final.get_node(node_id).data.get_time())
            # print(parent_id, node_id)

    for node_id in cell_tree_final.expand_tree(sorting=False):

        if node_id != 'P0':
            # normalize frame to time (min)
            # around 150, there are a lot of cut off edge, some cells appear and then disappear immediately
            # which reduce to different embryo's average start tp and end tp would be very different
            start_time, end_time = min(cell_tree_final.get_node(node_id).data.get_time()) * time_frame_resolution, max(
                cell_tree_final.get_node(node_id).data.get_time()) * time_frame_resolution
            cell_tree_final.get_node(node_id).data.set_time(
                list(np.arange(start=int(start_time), stop=int(end_time), step=1)))

            parent_id = cell_tree_final.parent(node_id).identifier
            # print(parent_id, cell_tree_final.get_node(parent_id).data.get_time())
            # print(node_id, cell_tree_final.get_node(node_id).data.get_time())
            if parent_id != 'P0' and cell_tree_final.get_node(parent_id).data.get_time()[-1] + 1 != \
                    cell_tree_final.get_node(node_id).data.get_time()[0]:
                # print('the mother cell division time doesn\'t match daughters\' appearance time')
                # print(parent_id, 'start, end ', cell_tree_final.get_node(parent_id).data.get_time()[0],
                #       cell_tree_final.get_node(parent_id).data.get_time()[-1])
                # print(node_id, 'start, end ', cell_tree_final.get_node(node_id).data.get_time()[0],
                #       cell_tree_final.get_node(node_id).data.get_time()[-1])
                # make the tree time become continuous
                cell_tree_final.get_node(parent_id).data.set_time(
                    cell_tree_final.get_node(parent_id).data.get_time() + list(np.arange(
                        start=cell_tree_final.get_node(parent_id).data.get_time()[-1] + 1,
                        stop=cell_tree_final.get_node(node_id).data.get_time()[0], step=1)))
                # print(parent_id, 'start, end ', cell_tree_final.get_node(parent_id).data.get_time()[0],
                #       cell_tree_final.get_node(parent_id).data.get_time()[-1])

    return cell_tree_final, begin_frame

    # 在这里开始，begin time, end time for every cell!, two daughter cell should have same begin time!
    # begin time for each cell 完成后，就有了frame的图，有了frame，就有了time的树，有了time，通过time找frame，所有embryo对应frame在那个time的平均，


def construct_basic_cell_name_tree(cd_file_path, max_time, tree_distance_num, name_dictionary_path):
    """
    Construct cell tree structure with cell names
    :param cd_file_path:  the name list file to the tree initilization
    :param max_time: the maximum time point to be considered
    :param name_dictionary_path: name dictionary of cell labels
    :return cell_tree: cell tree structure where each time corresponds to one cell (with specific name)
    """

    # read and combine all names from different acetrees
    # Get cell number by its name
    label_name_dict = pd.read_csv(name_dictionary_path, index_col=0).to_dict()['0']
    name_label_dict = {value: key for key, value in label_name_dict.items()}
    #  Construct cell
    #  Add irregular naming
    # initialize the cell tree (basic cell tree -- ABa, E MS, C, Z3 Z2 ,etc)
    cell_tree = Tree()
    cell_tree.create_node('P0', 'P0',
                          data=cell_node_info(number=0, time=[], generation_num=-1, position_x=0))
    cell_tree.create_node('AB', 'AB', parent='P0',
                          data=cell_node_info(number=name_label_dict['AB'], time=[], generation_num=0,
                                              position_x=-2 ** tree_distance_num))
    P1_CELL_NODE = cell_tree.create_node('P1', 'P1', parent='P0',
                                         data=cell_node_info(number=name_label_dict['P1'], time=[], generation_num=0,
                                                             position_x=2 ** tree_distance_num))
    EMS_CELL_NODE = cell_tree.create_node('EMS', 'EMS', parent='P1',
                                          data=cell_node_info(number=name_label_dict['EMS'], time=[], generation_num=1,
                                                              position_x=P1_CELL_NODE.data.get_position_x() - 2 ** (
                                                                      tree_distance_num - 1)))
    # MS,E daughters of EMS
    cell_tree.create_node('MS', 'MS', parent='EMS',
                          data=cell_node_info(number=name_label_dict['MS'], time=[], generation_num=2,
                                              position_x=EMS_CELL_NODE.data.get_position_x() - 2 ** (
                                                      tree_distance_num - 2)))
    cell_tree.create_node('E', 'E', parent='EMS',
                          data=cell_node_info(number=name_label_dict['E'], time=[], generation_num=2,
                                              position_x=EMS_CELL_NODE.data.get_position_x() + 2 ** (
                                                      tree_distance_num - 2)))
    # P2
    P2_CELL_NODE = cell_tree.create_node('P2', 'P2', parent='P1',
                                         data=cell_node_info(number=name_label_dict['P2'], time=[], generation_num=1,
                                                             position_x=P1_CELL_NODE.data.get_position_x() + 2 ** (
                                                                     tree_distance_num - 1)))

    # C,P3 daughters of P2
    cell_tree.create_node('C', 'C', parent='P2',
                          data=cell_node_info(number=name_label_dict['C'], time=[], generation_num=2,
                                              position_x=P2_CELL_NODE.data.get_position_x() - 2 ** (
                                                      tree_distance_num - 2)))
    P3_CELL_NODE = cell_tree.create_node('P3', 'P3', parent='P2',
                                         data=cell_node_info(number=name_label_dict['P3'], time=[], generation_num=2,
                                                             position_x=P2_CELL_NODE.data.get_position_x() + 2 ** (
                                                                     tree_distance_num - 2)))
    # D, P4 daughters of P3
    cell_tree.create_node('D', 'D', parent='P3',
                          data=cell_node_info(number=name_label_dict['D'], time=[], generation_num=3,
                                              position_x=P3_CELL_NODE.data.get_position_x() - 2 ** (
                                                      tree_distance_num - 3)))
    P4_CELL_NODE = cell_tree.create_node('P4', 'P4', parent='P3',
                                         data=cell_node_info(number=name_label_dict['P4'], time=[], generation_num=3,
                                                             position_x=P3_CELL_NODE.data.get_position_x() + 2 ** (
                                                                     tree_distance_num - 3)))
    # Z3 Z2 daughters of P4
    cell_tree.create_node('Z3', 'Z3', parent='P4',
                          data=cell_node_info(number=name_label_dict['Z3'], time=[], generation_num=4,
                                              position_x=P4_CELL_NODE.data.get_position_x() - 2 ** (
                                                      tree_distance_num - 4)))
    cell_tree.create_node('Z2', 'Z2', parent='P4',
                          data=cell_node_info(number=name_label_dict['Z2'], time=[], generation_num=4,
                                              position_x=P4_CELL_NODE.data.get_position_x() + 2 ** (
                                                      tree_distance_num - 4)))

    # Read the name excel and construct the tree with complete segCell
    df_cell_CD_file = read_cd_file(cd_file_path)

    # =====================================
    # dynamic update the name dictionary
    # =====================================
    # cell_in_dictionary = list(cell_number_dict.keys())
    #
    # # erase the cell excced max time
    # ace_pd = df_cell_CD_file[df_cell_CD_file.time <= max_time]
    # cell_list = list(ace_pd.cell.unique())
    # add_cell_list = list(set(cell_list) - set(cell_in_dictionary))
    # # if embryo CD files are different from cell name csv file(dictionary)
    # assert len(add_cell_list) == 0, "Name dictionary should be updated"

    # ================================= cancel dynamic updating ============
    # add_cell_list.sort()
    # if len(add_cell_list) > 0:
    #     print("Name dictionary updated !!!")
    #     add_number_dictionary = dict(zip(add_cell_list, range(len(cell_in_dictionary) + 1, len(cell_in_dictionary) + len(add_cell_list) + 1)))
    #     number_dictionary.update(add_number_dictionary)
    #     pd_number_dictionary = pd.DataFrame.from_dict(number_dictionary, orient="index")
    #     pd_number_dictionary.to_csv('./dataset/number_dictionary.csv', header=False)

    df_cell_CD_file = df_cell_CD_file[df_cell_CD_file.time <= max_time]
    cells_this_CD_embryo = list(df_cell_CD_file.cell.unique())
    for cell_name in list(cells_this_CD_embryo):
        if cell_name not in name_label_dict.keys():  # the cell with no nucleus or generated by acetree unknown
            continue

        if not cell_tree.contains(cell_name):  # this cell not yet in the cell tree
            if "Nuc" not in cell_name:  # normal name of the cell
                parent_name = cell_name[:-1]
                this_cell_node = cell_tree.create_node(cell_name, cell_name, parent=parent_name,
                                                       data=cell_node_info(number=name_label_dict[cell_name], time=[]))
                parent_node = cell_tree.parent(cell_name)
                this_cell_node.data.set_generation(parent_node.data.get_generation() + 1)
                if len(cell_tree.children(parent_name)) == 1:
                    this_cell_node.data.set_position_x(parent_node.data.get_position_x() - 2 ** (
                            tree_distance_num - this_cell_node.data.get_generation()))
                else:
                    # len ==2
                    this_cell_node.data.set_position_x(parent_node.data.get_position_x() + 2 ** (
                            tree_distance_num - this_cell_node.data.get_generation()))

    return cell_tree


# ===========================================================================================


def read_new_cd(cd_file):
    df_nuc = pd.read_csv(cd_file, lineterminator="\n")
    df_nuc[["cell", "time"]] = df_nuc["Cell & Time"].str.split(":", expand=True)
    df_nuc = df_nuc.rename(columns={"X (Pixel)": "x", "Y (Pixel)": "y", "Z (Pixel)\r": "z"})
    df_nuc = df_nuc.astype({"x": float, "y": float, "z": float, "time": int})

    return df_nuc


def read_cd_file(cd_file):
    df_nuc = pd.read_csv(cd_file, lineterminator="\n")
    df_nuc = df_nuc.astype({"x": float, "y": float, "z": float, "time": int})

    return df_nuc


def construct_celltree(nucleus_file, max_time,name_ditionary_path):
    '''
    Construct cell tree structure with cell names
    :param nucleus_file:  the name list file to the tree initilization
    :param max_time: the maximum time point to be considered
    :return cell_tree: cell tree structure where each time corresponds to one cell (with specific name)
    '''

    ##  Construct cell
    #  Add unregulized naming
    cell_tree = Tree()
    cell_tree.create_node('P0', 'P0')
    cell_tree.create_node('AB', 'AB', parent='P0')
    cell_tree.create_node('P1', 'P1', parent='P0')
    cell_tree.create_node('EMS', 'EMS', parent='P1')
    cell_tree.create_node('P2', 'P2', parent='P1')
    cell_tree.create_node('P3', 'P3', parent='P2')
    cell_tree.create_node('C', 'C', parent='P2')
    cell_tree.create_node('P4', 'P4', parent='P3')
    cell_tree.create_node('D', 'D', parent='P3')
    cell_tree.create_node('Z2', 'Z2', parent='P4')
    cell_tree.create_node('Z3', 'Z3', parent='P4')

    # EMS
    cell_tree.create_node('E', 'E', parent='EMS')
    cell_tree.create_node('MS', 'MS', parent='EMS')

    # Read the name excel and construct the tree with complete segCell
    df_time = read_cd_file(nucleus_file)

    # read and combine all names from different acetrees
    ## Get cell number
    try:
        label_name_dict = pd.read_csv(name_ditionary_path, index_col=0).to_dict()['0']
        name_label_dict = {value: key for key, value in label_name_dict.items()}
    except:
        raise Exception("Not find number dictionary at ./dataset")

    df_time = df_time[df_time.time <= max_time]
    all_cell_names = list(df_time.cell.unique())
    for cell_name in list(all_cell_names):
        if cell_name not in name_label_dict:
            continue
        times = list(df_time.time[df_time.cell == cell_name])
        cell_info = cell_node()
        cell_info.set_number(name_label_dict[cell_name])
        cell_info.set_time(times)
        if not cell_tree.contains(cell_name):
            if "Nuc" not in cell_name:
                parent_name = cell_name[:-1]
                cell_tree.create_node(cell_name, cell_name, parent=parent_name, data=cell_info)
        else:
            cell_tree.update_node(cell_name, data=cell_info)

    return cell_tree


class cell_node(object):
    # Node Data in cell tree
    def __init__(self):
        self.number = 0
        self.time = 0

    def set_number(self, number):
        self.number = number

    def get_number(self):
        return self.number

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time


class cell_node_info(object):
    # Node Data in cell tree
    def __init__(self, number=0, time=[], generation_num=0, position_x=0):
        self.number = number
        self.time = time
        self.generation = generation_num
        self.position_x = position_x

    def set_number(self, number):
        self.number = number

    def get_number(self):
        return self.number

    def set_generation(self, generation_num):
        self.generation = generation_num

    def get_generation(self):
        return self.generation

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def set_position_x(self, position_x):
        self.position_x = position_x

    def get_position_x(self):
        return self.position_x
