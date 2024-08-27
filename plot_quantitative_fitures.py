import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
from treelib import Tree
import numpy as np
import pandas as pd
import math
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns


def plot_cellwise_exp_reproducibility_from_mulemb_fig6b():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']

    renaming_dict = {'200710hmr1plc1p1': 'MT_C_hmr-1_Sample1', '200710hmr1plc1p2': 'MT_C_hmr-1_Sample2',
                     '200710hmr1plc1p3': 'MT_C_hmr-1_Sample3'}
    max_times = [96, 100, 100]
    last_4_cell_tp = [8, 11, 4]
    longest_embryo_idx = 2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path,
                                          '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1],
                      cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1), desc='assembling expression of result'):
        real_time_this = (tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp = round(real_time_this / time_resolution + last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path = os.path.join(path_tmp,
                                               '{}_{}_cellwise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path, 'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_label_, cell_this_value in cellwise_expression_dict.items():
                    cell_name_ = label_name_dict[cell_label_]

                    if cell_this_value < 1:
                        thistp_expression_thisemb_thiscell = 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        thistp_expression_thisemb_thiscell = cell_this_value
                        # thistp_expression_thisemb_thiscell = cell_this_value / cellwise_morphology_dict[cell_label_]

                    real_time_min=round(real_time_this)
                    if str(real_time_min).zfill(3) + '::' + cell_name_ in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(real_time_min).zfill(3) + '::' + cell_name_][
                            embryo_name] = thistp_expression_thisemb_thiscell
                    else:
                        cellwise_expression_value_dict_allembs[str(real_time_min).zfill(3) + '::' + cell_name_] = {
                            embryo_name: thistp_expression_thisemb_thiscell}
    plotting_df = pd.DataFrame(columns=['Embryo Name', 'minute and cell name', 'avg exp', 'this exp','variation ratio'])
    for tp_cell_name, emb_exp_dict in cellwise_expression_value_dict_allembs.items():
        if len(emb_exp_dict) > 1:
            avg_exp = sum(emb_exp_dict.values()) / len(emb_exp_dict)
            for emb_name, exp_value in emb_exp_dict.items():
                plotting_df.loc[len(plotting_df)] = [renaming_dict[emb_name], tp_cell_name, avg_exp, exp_value,abs(avg_exp-exp_value)/avg_exp]
    plotting_df.to_csv('expression record.csv', index=False)

    plt.xlim(0, 1)
    sns.histplot(data=plotting_df, x='variation ratio', binwidth=.1)
    plt.savefig('expression_variation_fig.pdf', format='pdf')
    plt.cla()

    max_value = max(list(plotting_df['avg exp']) + list(plotting_df['this exp']))
    plt.plot([0, max_value], [0, max_value])

    sns.scatterplot(data=plotting_df, x='avg exp', y='this exp', hue='Embryo Name')


    ax = plt.gca()
    # use sci demical style
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    # plt.show()
    plt.savefig('expression_fig.pdf', format='pdf')


def plot_and_save_following_cellls_fig6c():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']

    cells_precedent = ['ABala', 'ABalp', 'ABara', 'ABarp',
                       'ABpla', 'ABplp', 'ABpra', 'ABprp',
                       'MSa', 'MSp',
                       'Ea', 'Ep'
                             'Ca', 'Cp',
                       'Da', 'Dp'
                             'Z3', 'Z2'
                       ]

    cells_precedent = ['ABalaa','ABalap' ,'ABalpa', 'ABalpp',
                       'ABaraa','ABarap', 'ABarpa','ABarp',
                       'ABplaa','ABplap', 'ABplpa','ABplpp',
                       'ABpraa','ABprap', 'ABprpa','ABprpp',
                       'MSaa','MSap', 'MSpa','MSpp',
                       'Eaa','Eap', 'Epa','Epp',
                        'Caa',     'Cap', 'Cpa','Cpp',
                       'Daa','Dap', 'Dpa','Dpp',
                             'Z3', 'Z2'
                       ]

    renaming_dict = {'200710hmr1plc1p1': 'MT_C_hmr-1_Sample1', '200710hmr1plc1p2': 'MT_C_hmr-1_Sample2',
                     '200710hmr1plc1p3': 'MT_C_hmr-1_Sample3'}
    max_times = [96, 100, 100]
    last_4_cell_tp = [8, 11, 4]
    longest_embryo_idx = 2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path,
                                          '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1],
                      cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1), desc='assembling expression of result'):
        real_time_this = (tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp = round(real_time_this / time_resolution + last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path = os.path.join(path_tmp,
                                               '{}_{}_surfacewise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path, 'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_label_, cell_this_value in cellwise_expression_dict.items():
                    cell_name_ = label_name_dict[cell_label_]

                    if cell_this_value < 1:
                        thistp_expression_thisemb_thiscell = 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        # thistp_expression_thisemb_thiscell = cell_this_value
                        thistp_expression_thisemb_thiscell = cell_this_value / cellwise_morphology_dict[cell_label_]

                    if cell_name_.startswith('AB') and len(cell_name_) >= 6:
                        cells_key = cell_name_[:6]
                    elif cell_name_.startswith('MS') and len(cell_name_) >= 4:
                        cells_key = cell_name_[:4]
                    elif ((cell_name_.startswith('E') and cell_name_ != 'EMS') or cell_name_.startswith(
                            'C') or cell_name_.startswith('D') ) and len(cell_name_) >= 3:
                        cells_key = cell_name_[:3]
                    elif cell_name_.startswith('Z') and len(cell_name_) >= 2:
                        cells_key = cell_name_[:2]

                    else:
                        continue

                    if cells_key not in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[cells_key] = {
                            embryo_name: [thistp_expression_thisemb_thiscell]}
                    elif embryo_name not in cellwise_expression_value_dict_allembs[cells_key].keys():
                        cellwise_expression_value_dict_allembs[cells_key][embryo_name] = [
                            thistp_expression_thisemb_thiscell]
                    else:
                        cellwise_expression_value_dict_allembs[cells_key][embryo_name].append(
                            thistp_expression_thisemb_thiscell)

    plotting_dot_df = pd.DataFrame(
        columns=['Embryo Name', 'Cell Name Lineage', 'Average Expression', 'Cumulative Expression'])

    calculating_term='Average Expression'
    avg_curve_dict={}
    for cell_name, emb_exp_dict in cellwise_expression_value_dict_allembs.items():
        tem_list=[]
        for emb_name, emb_cell_list in emb_exp_dict.items():
            sum_exp = sum(emb_cell_list)
            avg_exp = sum_exp / len(emb_cell_list)
            plotting_dot_df.loc[len(plotting_dot_df)] = [renaming_dict[emb_name], cell_name, avg_exp, sum_exp]
            if calculating_term=='Cumulative Expression':
                tem_list.append(sum_exp)
            elif calculating_term=='Average Expression':
                tem_list.append(avg_exp)
            else:
                print('ERROR CAUSED')
        avg_curve_dict[cell_name]=sum(tem_list)/len(tem_list)

    plt.plot(list(avg_curve_dict.keys()),list(avg_curve_dict.values()),color='black',linestyle='dashed')

    plotting_dot_df.to_csv('expression record.csv', index=False)
    sns.scatterplot(data=plotting_dot_df, x='Cell Name Lineage', y=calculating_term, hue='Embryo Name')

    plt.xticks(rotation=75)

    # plt.show()
    plt.savefig('expression_fig.pdf', format='pdf')

def plot_and_save_following_cellls_fig6d():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']
    cells_precedent = ['124::ABprppaa',
                       '124::Caap',
                       '122::MSpppa',
                       '122::Epl'
                             '122::Dp',
                             'P4'
                       ]

    renaming_dict = {'200710hmr1plc1p1': 'MT_C_hmr-1_Sample1', '200710hmr1plc1p2': 'MT_C_hmr-1_Sample2',
                     '200710hmr1plc1p3': 'MT_C_hmr-1_Sample3'}
    max_times = [96, 100, 100]
    last_4_cell_tp = [8, 11, 4]
    longest_embryo_idx = 2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path,
                                          '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1],
                      cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1), desc='assembling expression of result'):
        real_time_this = (tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp = round(real_time_this / time_resolution + last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path = os.path.join(path_tmp,
                                               '{}_{}_surfacewise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path, 'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_label_, cell_this_value in cellwise_expression_dict.items():
                    cell_name_ = label_name_dict[cell_label_]
                    # calculating asymmetries
                    if (cell_name_.startswith('AB') and len(cell_name_) >= 4) or \
                            (cell_name_.startswith('MS') and len(cell_name_) >= 3) or \
                            (cell_name_.startswith('E') and cell_name_ != 'EMS' and len(cell_name_)>=2) or \
                            (cell_name_.startswith('C') and len(cell_name_)>=2) or \
                            (cell_name_.startswith('D') and len(cell_name_)>=2) or \
                            (cell_name_.startswith('Z') and len(cell_name_)>=2):
                        pass
                    else:
                        continue

                    if cell_this_value < 1:
                        thistp_expression_thisemb_thiscell = 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        # thistp_expression_thisemb_thiscell = cell_this_value
                        thistp_expression_thisemb_thiscell = cell_this_value / cellwise_morphology_dict[cell_label_]

                    real_int_min=round(real_time_this)
                    if str(real_int_min).zfill(3)+'::'+cell_name_ not in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(real_int_min).zfill(3)+'::'+cell_name_] = {
                            embryo_name: thistp_expression_thisemb_thiscell}
                    else:
                        cellwise_expression_value_dict_allembs[str(real_int_min).zfill(3)+'::'+cell_name_][embryo_name]=thistp_expression_thisemb_thiscell
    tmp_avg_dict={}
    for min_cell_name, emb_exp_dict in cellwise_expression_value_dict_allembs.items():
        if len(emb_exp_dict)==3:
            sum_exp = sum(emb_exp_dict.values())
            avg_exp = sum_exp / len(emb_exp_dict)
            tmp_avg_dict[min_cell_name]=avg_exp

    asymmetric_cell_exp_dict={}
    asymmetric_cell_exp_pd=pd.DataFrame(columns=['Calculating TP','Cell Name','Asymmetry'])
    # *a - *p) / *a or *l - *r) /*l ---- positive means  a is big, negative mean p is big
    done_tp_cell_list=[]
    for min_cell_name, this_exp in tmp_avg_dict.items():
        this_tp_min,this_cell_name=min_cell_name.split('::')
        last_suffix=this_cell_name[-1]
        the_mother_cell_name=this_cell_name[:-1]
        if this_tp_min+'::'+the_mother_cell_name in done_tp_cell_list:
            continue

        if last_suffix=='a':
            another_last_suffix='p'
            is_a_or_l=True
        elif last_suffix=='p':
            another_last_suffix='a'
            is_a_or_l=False
        elif last_suffix=='l':
            another_last_suffix='r'
            is_a_or_l=True
        elif last_suffix=='r':
            another_last_suffix='l'
            is_a_or_l=False
        else:
            print('ERROR IN LAST SUFFIX') # program need to error

        sister_cell_name =the_mother_cell_name+another_last_suffix
        if this_tp_min+'::'+sister_cell_name in tmp_avg_dict:
            sister_exp=tmp_avg_dict[this_tp_min+'::'+sister_cell_name]
        else:
            continue

        done_tp_cell_list.append(this_tp_min+'::'+the_mother_cell_name)
        if is_a_or_l:
            asy_this=(this_exp-sister_exp)/this_exp
        else:
            asy_this=(sister_exp-this_exp)/sister_exp

        if the_mother_cell_name not in asymmetric_cell_exp_dict.keys():
            asymmetric_cell_exp_dict[the_mother_cell_name]=[asy_this]
        else:
            asymmetric_cell_exp_dict[the_mother_cell_name].append(asy_this)
        asymmetric_cell_exp_pd.loc[len(asymmetric_cell_exp_pd)]=[int(this_tp_min),the_mother_cell_name,asy_this]
    asymmetric_cell_exp_pd.to_csv('all_exp_asymmetric_info.csv',index=False)

    asymmetric_cell_exp_pd_avg_for_sorting=pd.DataFrame(columns=['Cell Name','Mean Asymmetry'])
    average_dict={}
    for cell_name_tmp,tpwise_list in asymmetric_cell_exp_dict.items():
        if len(tpwise_list)>=3:
            avg_tmp=np.mean(np.abs(np.array(tpwise_list)))
            asymmetric_cell_exp_pd_avg_for_sorting.loc[len(asymmetric_cell_exp_pd_avg_for_sorting)]=[cell_name_tmp,avg_tmp]
            average_dict[cell_name_tmp]=avg_tmp
    asymmetric_cell_exp_pd_avg_for_sorting.to_csv('expression_fig.csv',index=False)

    sorted_expression_dict={k: v for k, v in sorted(average_dict.items(), key=lambda item: item[1],reverse=True)}
    asymmetric_pd_plotting=asymmetric_cell_exp_pd[asymmetric_cell_exp_pd['Cell Name'].isin(list(sorted_expression_dict.keys())[:10])]
    max_expression = np.max(np.abs(np.array(asymmetric_pd_plotting['Asymmetry'])))*1.1
    asymmetric_pd_plotting.to_csv('test.csv')

    plt.ylim(-max_expression,max_expression)
    plt.plot([-6, 125], [0, 0], linestyle='dotted',c='black')
    plt.xlim(-6, 125)
    print(max(asymmetric_pd_plotting['Calculating TP']))
    # plt.xlim(-max_expression,max_expression)
    #
    # x_ticks_list=[0,20,40,60,80,100,120]
    # plt.xticks(x_ticks_list)

    ax=sns.lineplot(data=asymmetric_pd_plotting, x='Calculating TP', y='Asymmetry', hue='Cell Name',style='Cell Name',markers=True,dashes=False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, ncol=2)



    plt.savefig('expression_fig.pdf', format='pdf')

def plot_and_save_following_cellls_fig6e():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']

    renaming_dict = {'200710hmr1plc1p1': 'MT_C_hmr-1_Sample1', '200710hmr1plc1p2': 'MT_C_hmr-1_Sample2',
                     '200710hmr1plc1p3': 'MT_C_hmr-1_Sample3'}
    max_times = [96, 100, 100]
    last_4_cell_tp = [8, 11, 4]
    longest_embryo_idx = 2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path,
                                          '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1],
                      cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1), desc='assembling expression of result'):
        real_time_this = (tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp = round(real_time_this / time_resolution + last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path = os.path.join(path_tmp,
                                               '{}_{}_contactwise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path, 'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_contact.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_cell_contact_key, contact_this_value in cellwise_expression_dict.items():
                    cell1_label, cell2_label = cell_cell_contact_key.split('_')
                    cell1_name = label_name_dict[int(cell1_label)]
                    cell2_name = label_name_dict[int(cell2_label)]
                    this_contact_Namekey = cell1_name + '_' + cell2_name
                    another_contact_Namekey = cell2_name + '_' + cell1_name

                    if contact_this_value < 1:
                        thistp_expression_thisemb_thiscell = 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        thistp_expression_thisemb_thiscell = contact_this_value
                        # thistp_expression_thisemb_thiscell = contact_this_value / cellwise_morphology_dict[cell_cell_contact_key]
                    # ==========================================time point wise=======================================
                    minute_tp_int=round(real_time_this)
                    if str(minute_tp_int).zfill(3) + '::' + this_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(minute_tp_int).zfill(3) + '::' + this_contact_Namekey][
                            embryo_name] = thistp_expression_thisemb_thiscell
                    elif str(minute_tp_int).zfill(3) + '::' + another_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(minute_tp_int).zfill(3) + '::' + another_contact_Namekey][
                            embryo_name] = thistp_expression_thisemb_thiscell
                    else:
                        cellwise_expression_value_dict_allembs[str(minute_tp_int).zfill(3) + '::' + this_contact_Namekey] = {
                            embryo_name: thistp_expression_thisemb_thiscell}
                    # ================================================================================================
    # ================================time point wise==========================================
    plotting_df = pd.DataFrame(columns=['Embryo Name', 'tp and cell name', 'avg exp', 'this exp','variation ratio'])
    for tp_cell_name, emb_exp_dict in cellwise_expression_value_dict_allembs.items():
        if len(emb_exp_dict) > 1:
            avg_exp = sum(emb_exp_dict.values()) / len(emb_exp_dict)
            for emb_name, exp_value in emb_exp_dict.items():
                plotting_df.loc[len(plotting_df)] = [renaming_dict[emb_name], tp_cell_name, avg_exp,
                                                     exp_value,abs(avg_exp-exp_value)/avg_exp]
    plotting_df.to_csv('expression record.csv', index=False)

    plt.xlim(0, 1)
    sns.histplot(data=plotting_df, x='variation ratio', binwidth=.1)
    plt.savefig('expression_variation_fig.pdf', format='pdf')
    plt.cla()

    max_value = max(list(plotting_df['avg exp']) + list(plotting_df['this exp']))
    plt.plot([0, max_value], [0, max_value])

    sns.scatterplot(data=plotting_df, x='avg exp', y='this exp', hue='Embryo Name')
    # ====================================================================================

                    # ===============================cell-cell contact life cycle======================================
                    # if this_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                    #     if embryo_name in cellwise_expression_value_dict_allembs[this_contact_Namekey].keys():
                    #         cellwise_expression_value_dict_allembs[this_contact_Namekey][
                    #             embryo_name].append(thistp_expression_thisemb_thiscell)
                    #     else:
                    #         cellwise_expression_value_dict_allembs[this_contact_Namekey][
                    #             embryo_name] = [thistp_expression_thisemb_thiscell]
                    # elif another_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                    #     if embryo_name in cellwise_expression_value_dict_allembs[another_contact_Namekey].keys():
                    #         cellwise_expression_value_dict_allembs[another_contact_Namekey][
                    #             embryo_name].append(thistp_expression_thisemb_thiscell)
                    #     else:
                    #         cellwise_expression_value_dict_allembs[another_contact_Namekey][
                    #             embryo_name] = [thistp_expression_thisemb_thiscell]
                    # else:
                    #     cellwise_expression_value_dict_allembs[this_contact_Namekey] = {
                    #         embryo_name: [thistp_expression_thisemb_thiscell]}
                    # ==========================================================================================

    # ========================================================================================
    # plotting_df = pd.DataFrame(columns=['Embryo Name', 'tp and contact name', 'avg exp', 'this exp','variation ratio'])
    # for cell_name_contact, emb_exp_dict in cellwise_expression_value_dict_allembs.items():
    #     if len(emb_exp_dict) > 1:
    #         all_emb_list = []
    #         for emb_name, exp_value_list in emb_exp_dict.items():
    #             all_emb_list = all_emb_list + exp_value_list
    #         emb_avg_emb = sum(all_emb_list) / len(all_emb_list)
    #
    #         for emb_name, exp_value_list in emb_exp_dict.items():
    #             cell_cycle_exp = sum(exp_value_list) / len(exp_value_list)
    #
    #             plotting_df.loc[len(plotting_df)] = [renaming_dict[emb_name], cell_name_contact, emb_avg_emb,
    #                                                  cell_cycle_exp, abs(cell_cycle_exp - emb_avg_emb)/emb_avg_emb]
    # plotting_df.to_csv('expression record.csv', index=False)
    #
    # # plt.xlim(0,1)
    # # sns.histplot(data=plotting_df,x='variation ratio',binwidth=.1)
    # # plt.savefig('expression_variation_fig.pdf', format='pdf')
    # # plt.cla()
    #
    # max_value = max(list(plotting_df['avg exp']) + list(plotting_df['this exp']))
    # plt.plot([0, max_value], [0, max_value])
    #
    # sns.scatterplot(data=plotting_df, x='avg exp', y='this exp', hue='Embryo Name')
    # =========================================================================================

    ax = plt.gca()
    # use sci demical style
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    # plt.show()
    plt.savefig('expression_fig.pdf', format='pdf')

def plot_and_save_following_cellls_fig6f():
    lineage_tree_path = r'./Data/lineage_tree'
    stat_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\ExpressionStat'
    volume_surface_contact_file_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\CTransformer embryos segmentation\Statistics'

    embryo_names = ['200710hmr1plc1p1', '200710hmr1plc1p2', '200710hmr1plc1p3']

    renaming_dict = {'200710hmr1plc1p1': 'MT_C_hmr-1_Sample1', '200710hmr1plc1p2': 'MT_C_hmr-1_Sample2',
                     '200710hmr1plc1p3': 'MT_C_hmr-1_Sample3'}
    max_times = [96, 100, 100]
    last_4_cell_tp = [8, 11, 4]
    longest_embryo_idx = 2
    time_resolution = 1.39

    name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']

    this_lineage_tree_path = os.path.join(lineage_tree_path,
                                          '{}_cell_life_tree'.format(embryo_names[longest_embryo_idx]))
    with open(this_lineage_tree_path, 'rb') as f:
        # print(f)
        cell_life_tree = Tree(pkl.load(f))

    begin_frame = max(cell_life_tree.get_node('ABa').data.get_time()[-1],
                      cell_life_tree.get_node('ABp').data.get_time()[-1])

    cellwise_expression_value_dict_allembs = {}
    for tp_anchor in tqdm(range(1, max_times[longest_embryo_idx] + 1), desc='assembling expression of result'):
        real_time_this = (tp_anchor - begin_frame) * time_resolution
        for emb_idx, embryo_name in enumerate(embryo_names):

            tp = round(real_time_this / time_resolution + last_4_cell_tp[emb_idx])
            path_tmp = os.path.join(stat_file_root, embryo_name)
            this_emb_value_path = os.path.join(path_tmp,
                                               '{}_{}_contactwise_expression.txt'.format(embryo_name, str(tp).zfill(3)))
            if os.path.exists(this_emb_value_path):
                with open(this_emb_value_path, 'rb') as handle:
                    cellwise_expression_dict = pkl.load(handle)

                path_tmp = os.path.join(volume_surface_contact_file_root, embryo_name)
                with open(os.path.join(path_tmp, '{}_{}_contact.txt'.format(embryo_name, str(tp).zfill(3))),
                          'rb') as handle:
                    cellwise_morphology_dict = pkl.load(handle)

                for cell_cell_contact_key, contact_this_value in cellwise_expression_dict.items():
                    cell1_label, cell2_label = cell_cell_contact_key.split('_')
                    cell1_name = label_name_dict[int(cell1_label)]
                    cell2_name = label_name_dict[int(cell2_label)]
                    this_contact_Namekey = cell1_name + '_' + cell2_name
                    another_contact_Namekey = cell2_name + '_' + cell1_name

                    if contact_this_value < 1:
                        thistp_expression_thisemb_thiscell = 0
                    else:
                        # thistp_expression_thisemb_thiscell = (math.log(cell_this_value))
                        # thistp_expression_thisemb_thiscell = contact_this_value
                        thistp_expression_thisemb_thiscell = contact_this_value / cellwise_morphology_dict[cell_cell_contact_key]
                    # ==========================================time point wise=======================================
                    # time_min_real=int(real_time_this)
                    if str(real_time_this).zfill(3) + '::' + this_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(real_time_this).zfill(3) + '::' + this_contact_Namekey].append(
                            thistp_expression_thisemb_thiscell)
                    elif str(real_time_this).zfill(3) + '::' + another_contact_Namekey in cellwise_expression_value_dict_allembs.keys():
                        cellwise_expression_value_dict_allembs[str(real_time_this).zfill(3) + '::' + another_contact_Namekey].\
                            append(thistp_expression_thisemb_thiscell)
                    else:
                        cellwise_expression_value_dict_allembs[str(real_time_this).zfill(3) + '::' + this_contact_Namekey] = \
                            [thistp_expression_thisemb_thiscell]
                    # ================================================================================================

    # ================================time point wise==========================================
    contact_expression_record_dict={}
    plotting_df = pd.DataFrame(columns=['Fixed Minute','Cell-cell Contact', 'Embryo-wise Average'])
    for tp_contact_name, emb_exp_list in cellwise_expression_value_dict_allembs.items():
        if len(emb_exp_list) > 2:
            tp_min_this,contact_pair=tp_contact_name.split('::')
            emb_avg_exp = sum(emb_exp_list) / len(emb_exp_list)
            plotting_df.loc[len(plotting_df)] = [np.float32(tp_min_this), contact_pair, emb_avg_exp]
            if contact_pair in contact_expression_record_dict:
                contact_expression_record_dict[contact_pair].append(emb_avg_exp)
            else:
                contact_expression_record_dict[contact_pair]=[emb_avg_exp]
    plotting_df.to_csv('expression record.csv', index=False)

    # sns.scatterplot(data=plotting_df, x='Fixed Minute', y='Embryo-wise Average', hue='Cell-cell Contact',s=6,legend=False)
    # ax = plt.gca()
    # # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # plt.savefig('expression_fig.pdf', format='pdf')
    # plt.cla()

    counting_most_expression={}
    for contact_pair,expression_list in contact_expression_record_dict.items():
        if len(expression_list)>2:
            counting_most_expression[contact_pair]=sum(expression_list)/len(expression_list)
    sorted_expression_dict={k: v for k, v in sorted(counting_most_expression.items(), key=lambda item: item[1],reverse=False)}
    lowest_and_highest_5_pairs=list(sorted_expression_dict.keys())[:5]+list(sorted_expression_dict.keys())[-5:]
    plotting_df_this=plotting_df[plotting_df['Cell-cell Contact'].isin(lowest_and_highest_5_pairs)]

    sns.scatterplot(data=plotting_df_this, x='Fixed Minute', y='Embryo-wise Average', hue='Cell-cell Contact')
    ax = plt.gca()
    # use sci demical style
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.savefig('expression_fig.pdf', format='pdf')
    plt.cla()



if __name__ == "__main__":
    plot_and_save_following_cellls_fig6d()
