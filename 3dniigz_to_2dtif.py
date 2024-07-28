
import os
import multiprocessing as mp
from tqdm import tqdm

from Utils.niigz3d_to_tiff2d import seperate_3dniigz_to_2dtif


if __name__=="__main__":

    # embryo_names=['181210plc1p2','200326plc1p3','200326plc1p4']
    # embryo_names=['181210plc1p2']
    embryo_names=['200710hmr1plc1p1','200710hmr1plc1p2','200710hmr1plc1p3']


    maxtimes=[100,100,100]
    raw_xyzs=[(512,712,68),(512,712,68),(512,712,68)]
    # generativeNuc_dir_tem=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\m2nGAN\Running\prediction'
    generativeNuc_dir_tem=r'C:\Users\zelinli6\Downloads\TEMP'

    saving_out_raw_root=r'F:\TifsForLineage'

    for idx,embryo_name_this in enumerate(embryo_names):
        max_time=maxtimes[idx]
        x_raw,y_raw,z_raw=raw_xyzs[idx]
        # saving_out_raw_image_dir=os.path.join(saving_out_raw_root,embryo_name_this,'tif')
        saving_out_raw_image_dir=os.path.join(saving_out_raw_root,embryo_name_this,'tifR')

        mpPool = mp.Pool(min(mp.cpu_count() // 2, max_time))
        configs = []
        for tp in range(1, max_time + 1):
            niigz3d_this_path = os.path.join(generativeNuc_dir_tem,
                                             '{}_{}_rawMemb.nii.gz'.format(embryo_name_this, str(tp).zfill(3)))
            configs.append((niigz3d_this_path, saving_out_raw_image_dir, (x_raw, y_raw, z_raw)))

        for idx, _ in enumerate(tqdm(mpPool.imap_unordered(seperate_3dniigz_to_2dtif, configs), total=len(configs),
                                     desc="Enhancing Nucleus via DL of {}".format(embryo_name_this))):
            # self.segmentbarSignal.emit(9,10)
            pass