import pandas as pd
import os
import time
import numpy as np
from scipy import ndimage
from glob import glob
import pickle
from tqdm import tqdm
from treelib import Tree
import multiprocessing as mp

from Utils.data_io import nib_load, check_folder
from Utils.data_structure import read_cd_file
from lineage_stat.data_structure import construct_celltree



def save_expression_to_readable_file(name_dictionary_path, embryo_name, running_max_time, CD_file_root, stat_file_root):
    label_name_dict = pd.read_csv(name_dictionary_path, index_col=0).to_dict()['0']
    # name_label_dict = {value: key for key, value in label_name_dict.items()}

    # ---------------------read cd file---------------------------------
    cd_file_path = os.path.join(CD_file_root, 'CD{}.csv'.format(embryo_name))
    # cd_file_all_dict = {}
    cd_file_dataframe = read_cd_file(cd_file_path)
    # print('constructing cell tree for ', embryo_name)
    cell_tree = construct_celltree(cd_file_path, running_max_time, name_dictionary_path)
    # ---------------------------------------------------------------

    # -==============assemble ========== surface-------------and-----------volume===================================
    all_names = [cname for cname in cell_tree.expand_tree(mode=Tree.WIDTH)]
    # for idx, cell_name in enumerate(all_names):
    volume_embryo = pd.DataFrame(
        np.full(shape=(running_max_time, len(all_names)), fill_value=np.nan, dtype=np.float32),
        index=range(1, running_max_time + 1), columns=all_names)

    for tp in tqdm(range(1, running_max_time + 1),
                   desc='assembling volume and surface area of {} result'.format(embryo_name)):
        path_tmp = os.path.join(stat_file_root, embryo_name)
        with open(os.path.join(path_tmp, '{}_{}_cellwise_expression.txt'.format(embryo_name, str(tp).zfill(3))),
                  'rb') as handle:
            cellwise_expression_dict = pickle.load(handle)

        for cell_label_, vol_value in cellwise_expression_dict.items():
            cell_name_ = label_name_dict[cell_label_]
            volume_embryo.loc[tp, cell_name_] = vol_value.astype(np.float32)

    volume_embryo = volume_embryo.loc[:, ((volume_embryo != 0) & (~np.isnan(volume_embryo))).any(axis=0)]
    volume_embryo.to_csv(os.path.join(stat_file_root, embryo_name + '_cellwise_expression.csv'))

    # ------------contact-----------initialize the contact csv  file----------------------
    # Get tuble lists with elements from the list

    name_combination = []
    first_level_names = []
    for i, name1 in enumerate(all_names):
        for name2 in all_names[i + 1:]:
            if not (cell_tree.is_ancestor(name1, name2) or cell_tree.is_ancestor(name2, name1)):
                first_level_names.append(name1)
                name_combination.append((name1, name2))

    multi_index = pd.MultiIndex.from_tuples(name_combination, names=['cell1', 'cell2'])
    # print(multi_index)
    stat_embryo = pd.DataFrame(
        np.full(shape=(running_max_time, len(name_combination)), fill_value=np.nan, dtype=np.float32),
        index=range(1, running_max_time + 1), columns=multi_index)
    # set zero element to express the exist of the specific nucleus
    for cell_name in all_names:
        if cell_name not in first_level_names:
            continue
        try:
            cell_time = cell_tree.get_node(cell_name).data.get_time()
            cell_time = [x for x in cell_time if x <= running_max_time]
            stat_embryo.loc[cell_time, (cell_name, slice(None))] = 0
        except:
            cell_name
    # print(stat_embryo)
    # edges_view = point_embryo.edges(data=True)
    for tp in tqdm(range(1, running_max_time + 1),
                   desc='assembling contact surface of {} result'.format(embryo_name)):
        path_tmp = os.path.join(stat_file_root, embryo_name)
        with open(os.path.join(path_tmp, '{}_{}_contactwise_expression.txt'.format(embryo_name, str(tp).zfill(3))),
                  'rb') as handle:
            contact_expression_dict = pickle.load(handle)
        for contact_sur_idx, contact_sur_value in contact_expression_dict.items():
            [cell1, cell2] = contact_sur_idx.split('_')
            cell1_name = label_name_dict[int(cell1)]
            cell2_name = label_name_dict[int(cell2)]
            if (cell1_name, cell2_name) in stat_embryo.columns:
                stat_embryo.loc[tp, (cell1_name, cell2_name)] = contact_sur_value.astype(np.float32)
            elif (cell2_name, cell1_name) in stat_embryo.columns:
                stat_embryo.loc[tp, (cell2_name, cell1_name)] = contact_sur_value.astype(np.float32)
            else:
                pass
                # print('columns missing (cell1_name, cell2_name)')
    stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]
    # print(stat_embryo)
    stat_embryo.to_csv(os.path.join(stat_file_root, embryo_name + '_contactwise_expression.csv'))
    # --------------------------------------------------------------------------------------------


def calculate_volume_surface_contactwise_expression(config):
    embryo_seg_path = config[0]
    embryo_expression_root_path = config[1]
    saving_expression_data = config[2]

    embryo_name, time_point = os.path.basename(embryo_seg_path).split('_')[:2]
    embryo_expression_path = os.path.join(embryo_expression_root_path, embryo_name, 'Gene',
                                          '{}_{}_expression.nii.gz'.format(embryo_name, time_point))

    if embryo_seg_path is None:
        raise Exception("Sorry, no path_embryo given!")

    # ------------------------calculate surface points using dialation for each cell --------------------
    frame_this_embryo = str(time_point).zfill(3)
    file_name = embryo_name + '_' + frame_this_embryo + '.nii.gz'

    segmented_arr = nib_load(embryo_seg_path).astype(int)
    raw_expression_arr = nib_load(embryo_expression_path)
    # .transpose([2, 1, 0])

    start_time = time.time()

    # ======================cell contact pairs==========================

    # -----!!!!!!!!get cell edge boundary!!!!--------------
    cell_bone_mask = np.zeros(segmented_arr.shape)
    # start_time=time.time()
    for cell_idx in np.unique(segmented_arr)[1:]:
        this_cell_arr = (segmented_arr == cell_idx)
        this_cell_arr_dilation = ndimage.binary_dilation(this_cell_arr, iterations=1)
        cell_bone_mask[np.logical_xor(this_cell_arr_dilation, this_cell_arr)] = 1
    print('edge boundary timing', time.time() - start_time)

    # cell_mask = segmented_arr != 0
    # boundary_mask_tmp = np.logical_xor(cell_mask == 0 , ndimage.binary_dilation(cell_mask))
    # boundary_mask=np.zeros(segmented_arr.shape)
    # boundary_mask[boundary_mask_tmp]=1
    #
    # nib_save(cell_bone_mask,r'./test_1.nii.gz')
    # print((cell_bone_mask==1).sum())
    # nib_save(boundary_mask,r'./test_2.nii.gz')
    # print((boundary_mask==1).sum())

    [x_bound, y_bound, z_bound] = np.nonzero(cell_bone_mask)
    boundary_elements = []

    # find boundary/points between cells
    start_time = time.time()
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = segmented_arr[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))[1:]  # with order
        # print(neighbor_labels)
        # if 0 in neighbor_labels:
        #     neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:  # contact between two cells
            boundary_elements.append(sorted(neighbor_labels))
    print('getting contact pairs', time.time() - start_time)

    cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
    # ===================================================================

    volume_dict = {}
    surface_dict = {}
    contact_expression_dict = {}

    # =================contact wise==============================================
    start_time = time.time()
    for (label1, label2) in cell_contact_pairs:
        contact_mask_tmp = np.logical_and(ndimage.binary_dilation(segmented_arr == label1),
                                          ndimage.binary_dilation(segmented_arr == label2))
        contact_mask = np.logical_and(contact_mask_tmp, cell_bone_mask)
        contact_sum = contact_mask.sum()
        dilated_contact_mask = ndimage.binary_dilation(contact_mask)
        # print(dilated_contact_mask.sum())
        expression_sum = sum(raw_expression_arr[dilated_contact_mask])
        if contact_sum > 2:
            # cell_contact_pair_renew.append((label1, label2))
            str_key = str(label1) + '_' + str(label2)
            # contact_area_dict[str_key] = 0
            contact_expression_dict[str_key] = expression_sum
            print(str_key, expression_sum)
    print('get pair contact surface expression time', time.time() - start_time)
    # =============================================================================

    cell_list = np.unique(segmented_arr)

    for cell_key in cell_list:
        if cell_key != 0:
            dilated_cell_mask = ndimage.binary_dilation(segmented_arr == cell_key)
            expression_this_cell = sum(raw_expression_arr[dilated_cell_mask])
            volume_dict[cell_key] = expression_this_cell
            print(cell_key, expression_this_cell)

            this_cell_surface_mask = ndimage.binary_dilation(
                np.logical_xor(dilated_cell_mask, (segmented_arr == cell_key)))
            expression_this_cell_surface = sum(raw_expression_arr[this_cell_surface_mask])
            surface_dict[cell_key] = expression_this_cell_surface

    path_tmp = os.path.join(saving_expression_data, embryo_name)
    check_folder(path_tmp)
    with open(os.path.join(path_tmp, file_name.split('.')[0] + '_cellwise_expression.txt'), 'wb+') as handle:
        pickle.dump(volume_dict, handle, protocol=4)
    with open(os.path.join(path_tmp, file_name.split('.')[0] + '_surfacewise_expression.txt'), 'wb+') as handle:
        pickle.dump(surface_dict, handle, protocol=4)
    with open(os.path.join(path_tmp, file_name.split('.')[0] + '_contactwise_expression.txt'), 'wb+') as handle:
        pickle.dump(contact_expression_dict, handle, protocol=4)

    # -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # is_generate_cell_wise_gui_data= True # depulicate all raw images and segmented files for ITK-SNAP-CVE software

    # =====================calculate expression volume surface contact=====================================
    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']

    cell_identity_assigned_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation'
    cd_file_root_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\Nuclei_independent_cd_files'
    name_dictionary_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv'
    # cell_fate_dictionary = r'/home/cimda/INNERDisk1/ZELIN_MEMB_DATA/CMap CSahepr 34 packed membrane nucleus/CellFate.xls'

    specific_labeling_3D_path = r'E:\ProjectData\MembraneProject\AllRawDataPacked'

    stat_path = os.path.join(cell_identity_assigned_path, 'ExpressionStat')

    label_name_dict = pd.read_csv(name_dictionary_path, index_col=0).to_dict()['0']
    name_label_dict = {value: key for key, value in label_name_dict.items()}

    for idx, embryo_name in enumerate(embryo_names):
        this_embryo_segmented_files = sorted(
            glob(os.path.join(cell_identity_assigned_path, embryo_name, 'SegCell', '*nii.gz')))
        max_time = len(this_embryo_segmented_files)

        # ==================calculate the cell-wise and contact-wise expression value take long time==================
        configs = []

        for file_path in this_embryo_segmented_files:
            # config_tmp['time_point'] = tp
            configs.append([file_path, specific_labeling_3D_path, stat_path])
        mp_cpu_num = min(len(configs), mp.cpu_count() // 2 + 1)

        # mp_cpu_num = min(len(configs), 1)

        mpPool = mp.Pool(mp_cpu_num)
        print('total ', mp.cpu_count(), 'cpu, we create ', mp_cpu_num)
        for idx_, _ in enumerate(
                tqdm(mpPool.imap_unordered(calculate_volume_surface_contactwise_expression, configs),
                     total=len(this_embryo_segmented_files),
                     desc="calculating {} segmentations (cell and contact expression)".format(embryo_name))):
            pass
        # =============================================================================================

        # ===============================just group them into readable csv===============================
        save_expression_to_readable_file(name_dictionary_path, embryo_name, max_time,
                                         cd_file_root_path,
                                         # os.path.join(annotated_root_path, embryo_name),
                                         stat_path)
        # ===============================================================================================
    # =================================================================================================================
