
import os
import shutil
from time import time
import re
import argparse
# import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils.NiftiDataset import *
from tqdm import tqdm
import multiprocessing as mp


# def contour_distance(contour_label, d_threshold=15, threshold = 0.95):
#     contour_label = sitk.Cast(contour_label, sitk.sitkInt16)
#     # 计算 Danielsson 距离场
#     distance_map = sitk.SignedDanielssonDistanceMap(contour_label)
#     # 将距离场限制在阈值内
#     distance_map = sitk.RescaleIntensity(distance_map, outputMinimum=0.0, outputMaximum=float(d_threshold))
#     # 归一化距离场
#     norm_mem_edt = (float(d_threshold) - distance_map) / float(d_threshold)
#     # 选出大于 0.9 的像素
#     selected_pixels = sitk.BinaryThreshold(norm_mem_edt, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
#     return selected_pixels


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def Align(image, reference):
    image_array = sitk.GetArrayFromImage(image)
    label_origin = reference.GetOrigin()
    label_direction = reference.GetDirection()
    label_spacing = reference.GetSpacing()
    image = sitk.GetImageFromArray(image_array)
    image.SetOrigin(label_origin)
    image.SetSpacing(label_spacing)
    image.SetDirection(label_direction)
    return image


def CropBackground(image, label):
    size_new = (240, 240, 120)

    def Normalization(image):
        """
        Normalize an image to 0 - 255 (8bits)
        """
        normalizeFilter = sitk.NormalizeImageFilter()
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)
        image = normalizeFilter.Execute(image)  # set mean and std deviation
        image = resacleFilter.Execute(image)  # set intensity 0-255
        return image

    image2 = Normalization(image)
    label2 = Normalization(label)

    threshold = sitk.BinaryThresholdImageFilter()
    threshold.SetLowerThreshold(20)
    threshold.SetUpperThreshold(255)
    threshold.SetInsideValue(1)
    threshold.SetOutsideValue(0)

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

    image_mask = threshold.Execute(image2)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = np.transpose(image_mask, (2, 1, 0))

    import scipy
    centroid = scipy.ndimage.measurements.center_of_mass(image_mask)

    x_centroid = np.int(centroid[0])
    y_centroid = np.int(centroid[1])

    roiFilter.SetIndex([int(x_centroid - (size_new[0]) / 2), int(y_centroid - (size_new[1]) / 2), 0])

    label_crop = roiFilter.Execute(label)
    image_crop = roiFilter.Execute(image)

    return image_crop, label_crop


def Registration(image, label):
    image, image_sobel, label, label_sobel,  = image, image, label, label
    Gaus = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    image_sobel = Gaus.Execute(image_sobel)
    label_sobel = Gaus.Execute(label_sobel)
    fixed_image = label_sobel
    moving_image = image_sobel
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image,  sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    image = sitk.Resample(image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    return image, label


def deal_with_data(para):
    a=para[0]
    b=para[1]
    reference_input_image=para[2]
    reference_label_image=para[3]
    i=para[4]
    args=para[5]
    saving_path=para[6]

    save_directory_images = f'/home/cimda/zelinli/NucGAN/GAN/Data_folder/{saving_path}/images'
    save_directory_labels = f'/home/cimda/zelinli/NucGAN/GAN/Data_folder/{saving_path}/labels'

    if not os.path.isdir(save_directory_images):
        os.mkdir(save_directory_images)

    if not os.path.isdir(save_directory_labels):
        os.mkdir(save_directory_labels)

    # a = X_val[i]
    # b = Y_val[i]
    #
    # print(a, b)

    label = sitk.ReadImage(b)
    image = sitk.ReadImage(a, outputPixelType=sitk.sitkFloat32)  # must input float32
    # image = contour_distance(image, 2, 0.95)  # use distance from nuclei center to label nuclei
    # image = Align(image, reference_label_image)
    # label = Align(label, reference_input_image)
    try:
        image,label = Registration(image,label)
        # image = Align(image, reference_label_image)

        # image, _ = Registration(image, reference_input_image)
    except:
        print('failed to registration',a,b)
        return

    image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
    label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear')


    label_directory = os.path.join(str(save_directory_labels), str(i) + '.nii')
    image_directory = os.path.join(str(save_directory_images), str(i) + '.nii')
    sitk.WriteImage(image, image_directory)
    sitk.WriteImage(label, label_directory)




def running_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/T1',
                        help='path to the images a (Input)')
    parser.add_argument('--labels', default=r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/T2',
                        help='path to the images b (Target)')

    # parser.add_argument('--split', default=250, help='number of images for testing')
    parser.add_argument('--resolution', default=(1.6, 1.6, 1.6), help='new resolution to resample the all data')
    args = parser.parse_args()

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    reference_image = list_images[-1]  # setting a reference image to have all data in the same coordinate system
    reference_image = sitk.ReadImage(reference_image)
    reference_image = resample_sitk_image(reference_image, spacing=args.resolution, interpolator='linear')

    reference_label_image = list_labels[-1]  # setting a reference image to have all data in the same coordinate system
    reference_label_image = sitk.ReadImage(reference_label_image)
    reference_label_image = resample_sitk_image(reference_label_image, spacing=args.resolution, interpolator='linear')

    if not os.path.isdir(r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/train'):
        os.mkdir(r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/train')

    if not os.path.isdir(r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/test'):
        os.mkdir(r'/home/cimda/zelinli/NucGAN/GAN/Data_folder/test')

    assert len(list_images) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(list_images))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [list_images[i] for i in ind_val], [list_labels[i] for i in ind_val]
    X_trn, Y_trn = [list_images[i] for i in ind_train], [list_labels[i] for i in ind_train]
    print('number of images: %3d' % len(list_images))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    parameters=[]
    for i in range(len(X_val)):
        parameters.append([X_val[i], Y_val[i],reference_image,reference_label_image,i,args,'test'])

    mp_cpu_num = min(len(parameters), int(mp.cpu_count() - 10))
    print('all cpu process is ', mp.cpu_count(), 'we created ', mp_cpu_num)
    mpPool = mp.Pool(mp_cpu_num)
    for _ in tqdm(mpPool.imap_unordered(deal_with_data, parameters), total=len(parameters),
                  desc="{} registrating data, all cpu process is {}, we created {}".format(
                      'validating embryo',
                      str(mp.cpu_count()),
                      str(mp_cpu_num))):
        pass

    parameters=[]
    for i in range(len(X_trn)):
        parameters.append([X_trn[i], Y_trn[i], reference_image,reference_label_image, i, args,'train'])

    mp_cpu_num = min(len(parameters), int(mp.cpu_count() - 10))
    print('all cpu process is ', mp.cpu_count(), 'we created ', mp_cpu_num)
    mpPool = mp.Pool(mp_cpu_num)
    for _ in tqdm(mpPool.imap_unordered(deal_with_data, parameters), total=len(parameters),
                  desc="{} registrating data, all cpu process is {}, we created {}".format(
                      'training embryo',
                      str(mp.cpu_count()),
                      str(mp_cpu_num))):
        pass

if __name__ == "__main__":
    running_func()


