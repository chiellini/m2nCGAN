import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
from treelib import Tree
import numpy as np
import pandas as pd
import math
from matplotlib.colors import LinearSegmentedColormap, ListedColormap



def plot_and_save_avg_expression_tree_from_mulemb_fig6a():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']
    max_times = [96, 100, 100]
    last_4_cell_tp=[8,11,4]
    longest_embryo_idx=2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path, '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1], cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1),desc='assembling expression of result'):
        real_time_this=(tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp=round(real_time_this/time_resolution+last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path=os.path.join(path_tmp, '{}_{}_cellwise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path,'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_volume.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_label_, cell_this_value in cellwise_expression_dict.items():
                    cell_name_ = label_name_dict[cell_label_]

                    if cell_this_value < 1:
                         thistp_expression_thisemb_thiscell= 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        # thistp_expression_thisemb_thiscell = cell_this_value
                        thistp_expression_thisemb_thiscell = cell_this_value / cellwise_morphology_dict[cell_label_]
                    if str(tp).zfill(3) + '::' + cell_name_ in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(tp).zfill(3) + '::' + cell_name_].append(thistp_expression_thisemb_thiscell)
                    else:
                        cellwise_expression_value_dict_allembs[str(tp).zfill(3) + '::' + cell_name_]=[thistp_expression_thisemb_thiscell]

    cellwise_expression_value_dict={}
    for key_, value_list in cellwise_expression_value_dict_allembs.items():
        cellwise_expression_value_dict[key_]=sum(value_list)/len(value_list)

    colors2 = np.array(
        [
            # (178, 178, 178),(138,168,178),(98,138,178),(48,88,178),
            (20, 88, 178), (20, 97, 178), (24, 116, 205),
            (26, 122, 219), (28, 134, 238), (29, 144, 245), (30, 144, 255), (63, 180, 255), (89, 210, 255),

            (100, 255, 218), (128, 255, 178), (168, 255, 168),
            (188, 238, 104), (162, 205, 90), (255, 128, 80), (255, 69, 70), (255, 66, 66), (238, 66, 66),
            (205, 66, 66), (139, 66, 66),

            (66, 66, 66)
        ]) / 255

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
    draw_lineage_tree_with_values(cell_life_tree, values_dict=cellwise_expression_value_dict,
                                  plot_title='avg_embryo_wise', is_abs=False,
                                  color_map=cmap1,
                                  is_real_time=True, time_resolution=time_resolution, end_time_point=100,
                                  path_saving=r'./Data/lineage_figures')


def plot_single_embryo():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']
    max_times = [96, 100, 100]
    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    for emb_idx, embryo_name in enumerate(embryo_names):
        this_lineage_tree_path = os.path.join(lineage_tree_path, '{}_cell_life_tree'.format(embryo_name))
        with open(this_lineage_tree_path, 'rb') as f:
            # print(f)
            cell_life_tree = Tree(pkl.load(f))

        cellwise_expression_value_dict = {}
        for tp in tqdm(range(1, max_times[emb_idx] + 1),
                       desc='assembling volume and surface area of {} result'.format(embryo_name)):
            path_tmp = os.path.join(stat_file_root, embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_surfacewise_expression.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                cellwise_expression_dict = pkl.load(handle)

            path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                cellwise_morphology_dict = pkl.load(handle)

            for cell_label_, cell_this_value in cellwise_expression_dict.items():
                cell_name_ = label_name_dict[cell_label_]
                if cell_this_value < 1:
                    cellwise_expression_value_dict[str(tp).zfill(3) + '::' + cell_name_] = 0
                else:
                    # cellwise_expression_value_dict[str(tp).zfill(3)+'::'+cell_name_] = (math.log(cell_this_value))
                    # cellwise_expression_value_dict[str(tp).zfill(3)+'::'+cell_name_] = cell_this_value
                    cellwise_expression_value_dict[str(tp).zfill(3) + '::' + cell_name_] = cell_this_value / \
                                                                                           cellwise_morphology_dict[
                                                                                               cell_label_]

        # https: // www.webucator.com / article / python - color - constants - module /
        # colors = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'yellow2','yellow1','yellow2', 'lightblue1',lightblue', 'dodgerblue1',
        #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']

        # colors = np.array(
        #     [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
        #      (238, 238, 0), (255, 255, 0), (238, 238, 0),
        #      (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255
        # colors2 = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'darkolivegreen3','darkolivegreen2','darkolivegreen3', 'lightblue1',lightblue', 'dodgerblue1',
        #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']
        # colors2 = np.array(
        #     [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
        #      (162, 205, 90), (188, 238, 104), (162, 205, 90),
        #      (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255

        colors2 = np.array(
            [
                # (178, 178, 178),(138,168,178),(98,138,178),(48,88,178),
                (20, 88, 178), (20, 97, 178), (24, 116, 205),
                (26, 122, 219), (28, 134, 238), (29, 144, 245), (30, 144, 255), (63, 180, 255), (89, 210, 255),

                (100, 255, 218), (128, 255, 178), (168, 255, 168),
                (188, 238, 104), (162, 205, 90), (255, 128, 80), (255, 69, 70), (255, 66, 66), (238, 66, 66),
                (205, 66, 66), (139, 66, 66),

                (66, 66, 66)
            ]) / 255

        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
        draw_lineage_tree_with_values(cell_life_tree, values_dict=cellwise_expression_value_dict,
                                      plot_title=embryo_name, is_abs=False,
                                      color_map=cmap1,
                                      is_real_time=True, time_resolution=1.39, end_time_point=100,
                                      path_saving=r'./Data/lineage_figures')


def draw_lineage_tree_with_values(cell_tree: Tree, values_dict, plot_title='', color_map='seismic',
                                  is_real_time=False, time_resolution:float=1, is_abs=True, end_time_point=None, showing=False,
                                  path_saving=r'./Data/lineage_tree'):
    """

    :param cell_tree:
    :param values_dict:
    :param embryo_name:
    :return:
    """
    drawing_points_array = []
    if is_real_time and end_time_point:
        end_time_point = end_time_point * time_resolution
    # ABpl may appear 1 min later than ABal,we would set time 0 as ABa begin to split!
    # draw ABa, ABp, EMS, P1 only
    begin_frame = max(cell_tree.get_node('ABa').data.get_time()[-1], cell_tree.get_node('ABp').data.get_time()[-1])
    for node_id in cell_tree.expand_tree(sorting=False):
        this_cell_node = cell_tree.get_node(node_id)

        # -------------draw specify range cell lineage tree--------------------
        if len(this_cell_node.data.get_time()) == 0:
            continue
        time_int = this_cell_node.data.get_time()[0]
        if end_time_point:  # the end time points for the lineage tree is set
            if (is_real_time and (time_int - begin_frame) * time_resolution > end_time_point) or time_int > end_time_point:
                continue
        # --------------------------------------------------------------

        for queue_index, time_int in enumerate(this_cell_node.data.get_time()):
            tp_and_cell_index = f'{time_int:03}' + '::' + node_id

            # -------------draw specify range cell lineage tree--------------------
            if end_time_point:  # the end time points for the lineage tree is set
                if (is_real_time and (
                        time_int - begin_frame) * time_resolution > end_time_point) or time_int > end_time_point:
                    continue
            # --------------------------------------------------------------

            # print(values_dict)
            # print(this_cell_node.data.get_position_x(), time_int, values_dict[tp_and_cell_index])
            if tp_and_cell_index in values_dict.keys():
                # print(tp_and_cell_index,values_dict[tp_and_cell_index])
                if is_real_time:  # the tree is count with frame rather than time points
                    drawing_points_array.append(
                        [this_cell_node.data.get_position_x(), -(time_int - begin_frame) * time_resolution,
                         values_dict[tp_and_cell_index]])
                else:  # the time is available from the frame
                    drawing_points_array.append(
                        [this_cell_node.data.get_position_x(), -time_int, values_dict[tp_and_cell_index]])

                if queue_index == 0:
                    mother_position_x = cell_tree.parent(node_id).data.get_position_x()
                    for x in np.arange(min(mother_position_x, this_cell_node.data.get_position_x()),
                                       max(mother_position_x, this_cell_node.data.get_position_x())):
                        if is_real_time:
                            drawing_points_array.append(
                                [x, -(time_int - begin_frame) * time_resolution, values_dict[tp_and_cell_index]])
                        else:
                            drawing_points_array.append([x, -time_int, values_dict[tp_and_cell_index]])

    # ============NEED TO BE TUNED=====================
    # drawing_points_array.append([-9500, 0,14])
    # drawing_points_array.append([-9500, 0,6.4])

    # make yellow to becom the colorbar center
    np_drawing_points_array = np.array(drawing_points_array)
    # print(np.max(np_drawing_points_array[:, 2]),np.min(np_drawing_points_array[:, 2]),np.average(np_drawing_points_array[:, 2]),np.median(np_drawing_points_array[:, 2]))
    if is_abs:
        edge_value = np.nanmax(np.abs(np_drawing_points_array[:, 2]))
        # print(np.average(np.abs(np_drawing_points_array[:, 2])))
        # print(np.nanmax(np.abs(np_drawing_points_array[:, 2])))
        print('edge value', edge_value)
        drawing_points_array.append([0, 0, edge_value])
        drawing_points_array.append([0, 0, -edge_value])
    np_drawing_points_array = np.array(drawing_points_array)

    my_dpi = 100
    fig = plt.figure(figsize=(10000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    plt.scatter(x=np_drawing_points_array[:, 0], y=np_drawing_points_array[:, 1], marker='s', s=6,
                c=np_drawing_points_array[:, 2], cmap=color_map)
    # fig.colorbar(sc,ax=ax,)
    plt.axis('off')
    # plt.colorbar(sc,aspect=100,orientation="horizontal")
    # plt.title(plot_title, fontsize=80)

    # saving_path = os.path.join(data_path + r'lineage_tree/tree_plot', embryo_name)
    if not os.path.exists(path_saving):
        os.mkdir(path_saving)
    saving_path = os.path.join(path_saving, plot_title + '_avgsurface.pdf')
    # print(saving_path)

    cbar = plt.colorbar(location='bottom')
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs,fontsize=40)

    # time axis !
    # if not end_time_point:
        # plt.arrow(-8500,220, 0, -100, shape='full', lw=0, length_includes_head=True, head_width=5)
    plt.arrow(-8400, 20, 0, -168, width=20, shape='full', head_length=15)
    plt.text(-8650, -200, 'min', fontsize=60,font='Arial')
    plt.text(-8700, 0, '0', fontsize=60,font='Arial')
    plt.text(-8750, -50, '50', fontsize=60,font='Arial')
    plt.text(-8800, -100, '100', fontsize=60,font='Arial')
    plt.text(-8800, -150, '150', fontsize=60,font='Arial')

    if showing:
        plt.show()
    plt.savefig(saving_path, format='pdf')



if __name__ == "__main__":
    plot_and_save_avg_expression_tree_from_mulemb_fig6a()