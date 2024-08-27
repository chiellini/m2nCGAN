# run this file under its floder with : python *.py

# import dependency library

import sys
import os
import glob
from pickle import dump

import tqdm

import pandas as pd

# the python designer don't want to run file/scripts in the living inside module's directory,
# he consider it as antipattern, so we need to append this to make this script out of box.
# from numpy import average

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# import user defined library
from lineage_stat.data_structure import construct_basic_cell_name_tree

# data_path = r'D:/cell_shape_quantification/DATA/'

if __name__ == "__main__":
    # ================================
    # set parameters
    # ================================
    tree_distance_num = 12
    save_folder =  r'./Data/lineage_tree'
    cd_files_root_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\Nuclei_independent_cd_files'
    cell_info_folder_root=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics\NucLocFile'
    embryo_names = ['200710hmr1plc1p1','200710hmr1plc1p2','200710hmr1plc1p3']
    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"

    cell_size = []

    for embryo_name in embryo_names:
        cell_div_files_path = os.path.join(cd_files_root_path, "CD{}.csv".format(embryo_name))
        # Online folder
        cell_info_folder = os.path.join(cell_info_folder_root,embryo_name)
        save_name = None

        # number of tps
        # get all * .csv files list under the folder
        cell_info_files_path = sorted(glob.glob(os.path.join(cell_info_folder, "*.csv")))
        # print(cell_info_files)
        max_time = len(cell_info_files_path)
        # ================================
        # Construct cell label tree
        # ================================
        cell_tree = construct_basic_cell_name_tree(cd_file_path=cell_div_files_path, max_time=max_time,
                                                   tree_distance_num=tree_distance_num,
                                                   name_dictionary_path=name_dictionary_file_path)

        # cell_tree.show(key=False),

        # ================================
        # collect cell information (tps, volumns, surfaces)
        # ================================

        for tp, cell_info_file in enumerate(
                tqdm.tqdm(cell_info_files_path, desc="Collecting cell infos cshaper embryo{}".format(embryo_name)),
                start=1):
            if tp <= max_time:
                # go through each cell with time order
                df_cell_CD_and_note = pd.read_csv(cell_info_file, header=0).astype({"note": str, "nucleus_name": str})
                df_cell_CD_and_note = df_cell_CD_and_note[
                    ~df_cell_CD_and_note.note.str.contains("lost")]  # delete all lost cells
                df_cell_CD_and_note = df_cell_CD_and_note[
                    ~df_cell_CD_and_note.note.str.contains("child")]  # delete all children nucleus, but not cell.
                cells_at_this_tp = df_cell_CD_and_note["nucleus_name"].tolist()
                for cell_name in cells_at_this_tp:
                    this_cell_node = cell_tree.get_node(cell_name)
                    if this_cell_node.is_leaf():
                        this_cell_node.data.get_time().append(tp)
                    else:
                        [child1, child2] = cell_tree.children(cell_name)
                        if this_cell_node.is_leaf():
                            this_cell_node.data.get_time().append(tp)
                        elif len(child1.data.get_time()) == 0 and len(child1.data.get_time()) == 0:
                            this_cell_node.data.get_time().append(tp)
                        elif tp >= child1.data.get_time()[0] or tp >= child2.data.get_time()[0]:
                            continue
                        else:
                            this_cell_node.data.get_time().append(tp)
            else:
                break
        # for node_id in cell_tree.expand_tree(sorting=False):
        #     # if len(cell_tree.get_node(node_id).data.get_time())==0:
        #     this_cell_node = cell_tree.get_node(node_id)
        #     print(embryo_name, node_id, 'living tp', this_cell_node.data.get_time(), 'Generation number',
        #           this_cell_node.data.get_generation(), 'position_x', this_cell_node.data.get_position_x())
        # cell_size.append(cell_tree.size())
        # cell_tree.to_graphviz(embryo_name)
        # save cell tree incoporated with life span. Accessible with cell_tree.get_node[<cell_name>].data.get_time()

        # if cell_tree.get_node('ABal').data.get_time()[0] != cell_tree.get_node('ABpl').data.get_time()[0]:
        #     print('ABal and ABpl don\'t split together')
        #     print(cell_tree.get_node('ABal').data.get_time()[0])
        #     print(cell_tree.get_node('ABpl').data.get_time()[0])

        if save_name is None:
            save_name = os.path.basename(cell_info_folder)
        save_file = os.path.join(save_folder, save_name + "_cell_life_tree")
        with open(save_file, "wb") as f:
            dump(cell_tree, f)  # treelib tree object need pickle to dump and load

        begen_cell = ''
        for node_id in cell_tree.expand_tree(sorting=False):
            if len(cell_tree.get_node(node_id).data.get_time()) != 0:
                begen_cell = node_id
                break
        print(embryo_name,'  begin cell', begen_cell, ' cell number', cell_tree.size(),'embryo max frame', max_time)
