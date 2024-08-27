import os
import numpy as np

from skimage.transform import resize


from Utils.data_io import nib_save,nib_load

specific_labeling_3D_path=r'E:\ProjectData\MembraneProject\AllRawDataPacked'

embryo_names=[
    '200710hmr1plc1p1',
              # '200710hmr1plc1p2','200710hmr1plc1p3'
]
max_times=[
    100,
    # 100,100
]

for emb_idx,embryo_name in enumerate(embryo_names):
    for tp in range(1,max_times[emb_idx]+1):
        original_path=os.path.join(specific_labeling_3D_path,embryo_name,'RawNuc','{}_{}_rawNuc.nii.gz'.format(embryo_name,str(tp).zfill(3)))
        saving_path=os.path.join(specific_labeling_3D_path,embryo_name,'Gene','{}_{}_expression.nii.gz'.format(embryo_name,str(tp-4).zfill(3)))
        the_original_array=nib_load(original_path)
        the_saving_array=resize(image=the_original_array, output_shape=(256,356,160), preserve_range=True, order=1).astype(np.int16)
        nib_save(the_saving_array,saving_path)