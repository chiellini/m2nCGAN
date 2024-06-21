# import glob
import os.path

# import numpy as np
from skimage.morphology import ball
from scipy import ndimage
from skimage.transform import resize
from tqdm import tqdm
import multiprocessing as mp
from utils.NiftiDataset import *
from scipy.ndimage.morphology import distance_transform_edt


# from data_utils.augmentations import contour_distance
from utils.data_io import nib_load, nib_save

def contour_distance(contour_label, d_threshold=15):
    '''
    :param label:  Binary label of the target
    :param center_stack:
    :return:
    '''
    background_edt = distance_transform_edt(contour_label == 0) # calculate the distance from backgroud to labelmembrane
    # near -> far, small->large
    # print(np.unique(background_edt.astype(int), return_counts=True))

    background_edt[background_edt > d_threshold] = d_threshold
    norm_mem_edt = (d_threshold - background_edt) / d_threshold # the normalize distance from labelmembrane to backgroud

    return norm_mem_edt.astype(np.float32)


def generating_center_nucleus(para):
    niigz_this = para[0]
    saving_cell_path = para[1]
    is_ambiguous_here = para[2]

    embryo_name,tp = os.path.basename(niigz_this).split('.')[0].split('_')[:2]
    seg_cell_arr = nib_load(niigz_this)
    cell_list = np.unique(seg_cell_arr)[1:]
    new_memb_center = np.zeros(seg_cell_arr.shape)
    # new_memb_center[seg_cell_arr>0]=1
    # new_memb_center=ndimage.binary_erosion(new_memb_center,iterations=5)
    for cell_this_one in cell_list:
        # mask_this=(seg_cell_arr == cell_this_one)
        # volume_size=mask_this.sum()
        # if volume_size<15:
        #     continue
        # radius_this=int((volume_size*3/(4*3.14))**(1/3)+1)
        # iteration_this=radius_this-(6-int(int(tp)/100))
        # mask_this=ndimage.binary_erosion(mask_this,iterations=iteration_this)
        # new_memb_center[mask_this]=1
        this_cell_pos_list = np.where(seg_cell_arr == cell_this_one)
        center_pos = np.mean(this_cell_pos_list, axis=1)
        # center_x = this_cell_pos_list[0][center_idx]
        # center_y = this_cell_pos_list[1][center_idx]
        # center_z = this_cell_pos_list[2][center_idx]
        new_memb_center[int(center_pos[0]), int(center_pos[1]), int(center_pos[2])] = 1

    if is_ambiguous_here:
        nucleus_maker_footprint = ball(5 - int(int(tp) / 100))
        saving_nii = ndimage.grey_dilation(new_memb_center, footprint=nucleus_maker_footprint)
        image_path = os.path.join(saving_cell_path, '{}_{}_pseudoRawNuc.nii'.format(embryo_name,tp))
        nib_save(saving_nii.astype(np.float32), image_path)

        # fuzzy_memb_center = contour_distance(saving_nii, 3)
        image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkFloat32)  # must input float32
        # image = contour_distance(image, 2, 0.95)  # use distance from nuclei center to label nuclei
        # image = Align(image, reference_label_image)
        # label = Align(label, reference_input_image)


        image = resample_sitk_image(image, spacing=(1.6, 1.6, 1.6), interpolator='linear')
        # embryo_name, tp_this=os.path.basename(niigz_this).split('.')[0].split('_')[:2]
        # image_directory = os.path.join(saving_cell_path, '{}_{}_pseudoRawNuc.nii'.format(embryo_name,tp))
        sitk.WriteImage(image, image_path)

        # print(np.unique(fuzzy_memb_center))
        # image_resized = resize(
        #     fuzzy_memb_center, (int(fuzzy_memb_center.shape[0] * 0.625), int(fuzzy_memb_center.shape[1] * 0.625),
        #                         int(fuzzy_memb_center.shape[2] * 0.625)), anti_aliasing=True
        # )
        # frustration_arr=np.random.normal(0.9,0.1,image_resized.shape)
        # saving_nii=image_resized*frustration_arr
        # saving_nii[image_resized==1]=1
        # nib_save(saving_nii.astype(np.float32), os.path.join(saving_cell_path, os.path.basename(niigz_this)))

    else:
        nucleus_maker_footprint = ball(5 - int(int(tp) / 100))
        saving_nii = ndimage.grey_dilation(new_memb_center, footprint=nucleus_maker_footprint)
        nib_save(saving_nii, os.path.join(saving_cell_path, os.path.basename(niigz_this)))


def generate_input_for_m2nGAN_training_and_testing():
    training_embryos = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1']
    testing_embryos = ['200315plc1p2', '200326plc1p4']

    seg_cell_path = r'/home/cimda/INNERDisk1/ZELIN_MEMB_DATA/RunningDataset_4D_SWIN'
    saving_cell_path = r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/T1'

    for embryo_name in training_embryos:
        seg_cell_this = glob.glob(os.path.join(seg_cell_path, embryo_name, 'SegCell', '*.nii.gz'))
        parameters = []

        for niigz_this in seg_cell_this:
            parameters.append([niigz_this, saving_cell_path])

        mp_cpu_num = min(len(parameters), int(mp.cpu_count() - 10))
        print('all cpu process is ', mp.cpu_count(), 'we created ', mp_cpu_num)
        mpPool = mp.Pool(mp_cpu_num)
        for _ in tqdm(mpPool.imap_unordered(generating_center_nucleus, parameters), total=len(parameters),
                      desc="{} membrane --> nuclei, all cpu process is {}, we created {}".format(
                          'validating embryo',
                          str(mp.cpu_count()),
                          str(mp_cpu_num))):
            pass


def generate_input_for_m2nGAN_timelapse_evaluation():
    # training_embryos=['191108plc1p1','200109plc1p1','200113plc1p2','200113plc1p3','200322plc1p2','200323plc1p1']
    evaluation_embryos = ['181210plc1p2', '200326plc1p3', '200326plc1p4']
    # evaluation_embryos = ['200326plc1p4']


    seg_cell_path = r'/home/cimda/INNERDisk1/ZELIN_MEMB_DATA/RunningDataset_4D_SWIN'
    saving_cell_path = r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/Evaluation/images'

    for embryo_name in evaluation_embryos:
        seg_cell_this = glob.glob(os.path.join(seg_cell_path, embryo_name, 'SegCell', '*.nii.gz'))
        parameters = []

        for niigz_this in seg_cell_this:
            parameters.append([niigz_this, saving_cell_path, True])

        mp_cpu_num = min(len(parameters), int(mp.cpu_count() - 10))
        print('all cpu process is ', mp.cpu_count(), 'we created ', mp_cpu_num)
        mpPool = mp.Pool(mp_cpu_num)
        for _ in tqdm(mpPool.imap_unordered(generating_center_nucleus, parameters), total=len(parameters),
                      desc="{} membrane --> nuclei, all cpu process is {}, we created {}".format(
                          'validating embryo',
                          str(mp.cpu_count()),
                          str(mp_cpu_num))):
            pass


def generating_enhanced_gt_nucleus(para):
    raw_nuc_niigz = para[0]
    saving_cell_path = para[1]

    embryo_name, tp = os.path.basename(raw_nuc_niigz).split('.')[0].split('_')[:2]
    annotated_nuc_path = os.path.join(para[2], embryo_name, 'AnnotatedNuc',
                                      '{}_{}_annotatedNuc.nii.gz'.format(embryo_name, tp))

    raw_nuc_arr = nib_load(raw_nuc_niigz)
    annotated_nuc_arr = nib_load(annotated_nuc_path)

    raw_nuc_arr[annotated_nuc_arr > 0] = 240

    nib_save(raw_nuc_arr, os.path.join(saving_cell_path, os.path.basename(raw_nuc_niigz)))


def generate_gt_for_GAN():
    training_embryos = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1']
    testing_embryos = ['200315plc1p2', '200326plc1p4']

    raw_nuc_path = r'/home/cimda/zelinli/CellAtlas/DataSource/RunningDataset'
    saving_enhanced_nuc_path = r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/T2'

    for embryo_name in training_embryos:
        raw_nuclei_this = glob.glob(os.path.join(raw_nuc_path, embryo_name, 'RawNuc', '*.nii.gz'))
        # print(raw_nuclei_this)
        parameters = []

        for niigz_this in raw_nuclei_this:
            parameters.append([niigz_this, saving_enhanced_nuc_path, raw_nuc_path])

        mp_cpu_num = min(len(parameters), 3, int(mp.cpu_count() - 10))
        print('all cpu process is ', mp.cpu_count(), 'we created ', mp_cpu_num)
        mpPool = mp.Pool(mp_cpu_num)
        for _ in tqdm(mpPool.imap_unordered(generating_enhanced_gt_nucleus, parameters), total=len(parameters),
                      desc="{} raw nuc --> enhanced nuclei, all cpu process is {}, we created {}".format(
                          'validating embryo',
                          str(mp.cpu_count()),
                          str(mp_cpu_num))):
            pass


if __name__ == '__main__':
    generate_input_for_m2nGAN_timelapse_evaluation()
