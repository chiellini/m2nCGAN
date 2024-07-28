from skimage.transform import resize
from skimage.filters import gaussian
import numpy as np
import os
# from PIL import Image
import tifffile
from Utils.data_io import nib_load, check_folder


def seperate_3dniigz_to_2dtif(para):
    source_niigz_file_path, target_tif_root_path, raw_img_shape =para
    embryo_name,tp= os.path.basename(source_niigz_file_path).split('.')[0].split('_')[:2]
    generative_nuc_3d_volume=nib_load(source_niigz_file_path)
    generative_nuc_3d_volume=gaussian(generative_nuc_3d_volume,sigma=1.5)
    segmented_shape=generative_nuc_3d_volume.shape
    reshaped_3d_volume=resize(image=generative_nuc_3d_volume,output_shape=(raw_img_shape[0],raw_img_shape[1],segmented_shape[-1]),preserve_range=True,order=2,anti_aliasing=True).astype(np.uint8)
    for page_num in range(1, raw_img_shape[-1] + 1):

        raw_index= raw_img_shape[-1]+1 - page_num
        raw_index_in_generative=int(page_num * segmented_shape[-1] / raw_img_shape[-1])-1

        # print(raw_index,raw_index_in_generative)

        raw_tif_file_name='{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp).zfill(3),str(raw_index).zfill(2))
        saving_tif_path=os.path.join(target_tif_root_path, raw_tif_file_name)

        image_array_this_p=reshaped_3d_volume[:,:,raw_index_in_generative].astype(np.uint8)
        check_folder(saving_tif_path)

        # tif_image=Image.fromarray(image_array_this_p, mode="L")
        tifffile.imwrite(saving_tif_path, image_array_this_p,

                         byteorder='>',
                         # bigtiff=True,
                         imagej=True,
                         # ome=True,
                         resolution=(10.981529, 10.981529),
                         # 'info':{'compression':'raw'},
                         metadata={'size': (raw_img_shape[1], raw_img_shape[0]),
                                   'height': raw_img_shape[0],
                                   'width': raw_img_shape[1],
                                   'use_load_libtiff': False,
                                   'tile': [('raw', (0, 0, raw_img_shape[1], raw_img_shape[0]), 396, ('L', 0, 1))]})

        # tif_image.save(saving_tif_path,format='TIFF')

def seperate_3dniigz_to_2dtif_for_mem_and_nuc(para):
    source_niigz_file_path, target_tif_root_path, raw_img_shape =para
    embryo_name,tp= os.path.basename(source_niigz_file_path).split('.')[0].split('_')[:2]
    generative_nuc_3d_volume=nib_load(source_niigz_file_path)
    generative_nuc_3d_volume=gaussian(generative_nuc_3d_volume,sigma=1.5)
    segmented_shape=generative_nuc_3d_volume.shape
    reshaped_3d_volume=resize(image=generative_nuc_3d_volume,output_shape=(raw_img_shape[0],raw_img_shape[1],segmented_shape[-1]),preserve_range=True,order=2,anti_aliasing=True).astype(np.uint8)
    for page_num in range(1, raw_img_shape[-1] + 1):

        raw_index= raw_img_shape[-1]+1 - page_num
        raw_index_in_generative=int(page_num * segmented_shape[-1] / raw_img_shape[-1])-1

        # print(raw_index,raw_index_in_generative)

        raw_tif_file_name='{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp).zfill(3),str(raw_index).zfill(2))
        saving_tif_path=os.path.join(target_tif_root_path, raw_tif_file_name)

        image_array_this_p=reshaped_3d_volume[:,:,raw_index_in_generative].astype(np.uint8)
        check_folder(saving_tif_path)

        # tif_image=Image.fromarray(image_array_this_p, mode="L")
        tifffile.imwrite(saving_tif_path, image_array_this_p,

                         byteorder='>',
                         # bigtiff=True,
                         imagej=True,
                         # ome=True,
                         resolution=(10.981529, 10.981529),
                         # 'info':{'compression':'raw'},
                         metadata={'size': (raw_img_shape[1], raw_img_shape[0]),
                                   'height': raw_img_shape[0],
                                   'width': raw_img_shape[1],
                                   'use_load_libtiff': False,
                                   'tile': [('raw', (0, 0, raw_img_shape[1], raw_img_shape[0]), 396, ('L', 0, 1))]})
