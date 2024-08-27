import glob
import os
# import pickle
# import imageio
# import shutil
import numpy as np
# import pandas as pd
import nibabel as nib
from skimage.transform import resize


def nib_save(data, file_name):
    check_folder(file_name)
    return nib.save(nib.Nifti1Image(data, np.eye(4)), file_name)

def nib_load(file_name):
    if not os.path.exists(file_name):
        raise IOError("Cannot find file {}".format(file_name))
    return nib.load(file_name).get_fdata()

def check_folder(file_name):
    if "." in os.path.basename(file_name):
        dir_name = os.path.dirname(file_name)
    else:
        dir_name = file_name
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

prediction_root=r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/Running/Pred'
saving_resized_prediction_root=r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/Running/prediction'
max_tps={'200710hmr1plc1p1':100,'200710hmr1plc1p2':100,'200710hmr1plc1p3':100}
evaluation_embryos_and_sizes = {'200710hmr1plc1p1':(256,356,158), '200710hmr1plc1p2':(256,356,158), '200710hmr1plc1p3':(256,356,158)}

for embryo_name,this_embryo_resolution in evaluation_embryos_and_sizes.items():
    for tp_this in range(1,max_tps[embryo_name]+1):
        the_prediction_path_this=os.path.join(prediction_root,'{}_{}_pseudoRawNuc.nii'.format(embryo_name,str(tp_this).zfill(3)))
        the_prediction=nib_load(the_prediction_path_this)
        the_resized_prediction=resize(the_prediction,this_embryo_resolution,anti_aliasing=True)
        nib_save(the_resized_prediction.astype(np.uint8),os.path.join(saving_resized_prediction_root,'{}_{}_pseudoRawNuc.nii.gz'.format(embryo_name,str(tp_this).zfill(3))))

