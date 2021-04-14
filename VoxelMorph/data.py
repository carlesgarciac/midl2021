import os
import numpy as np
from nibabel.testing import data_path
from scipy import ndimage
import pandas as pd
import nibabel as nib
from torchvision import transforms
from PIL import Image
import pdb

# data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'
data_dir = '/home/carlesgc/Projects/midl2021/data_syn/Training/'

def data_import(data_dir):
    # data directory
    # data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'
    labeled_data_dir = os.path.join(data_dir, 'Labeled/')

    # Excel sheet loading
    xl_file = pd.ExcelFile(os.path.join(data_dir, 'M&Ms Dataset Information.xlsx'))
    dfs = {sheet_name: xl_file.parse(sheet_name)
            for sheet_name in xl_file.sheet_names}

    code = dfs['TrainingAnonUnlabeled']['External code']  # External Code
    ED = dfs['TrainingAnonUnlabeled']['ED']  # ED
    ES = dfs['TrainingAnonUnlabeled']['ES']  # ES

    # slices = []
    vols = []
    segs = []

    a, b, c = 0, 0, 0

    for patient in range(len(dfs['TrainingAnonUnlabeled']['External code'])):
        # Vendor C is not segmented
        if dfs['TrainingAnonUnlabeled']['Vendor'][patient] != 'C':
            # Load and convert to np the nii files
            load_nii_img = nib.load(os.path.join(labeled_data_dir, code[patient], code[patient]+'_sa.nii.gz'))
            load_nii_img = load_nii_img.get_fdata()
            load_nii_gt = nib.load(os.path.join(labeled_data_dir, code[patient], code[patient]+'_sa_gt.nii.gz'))
            load_nii_gt = load_nii_gt.get_fdata()

            # Get time frame with Ground truth
            volume_1ED = load_nii_img[:, :, :, ED[patient]]
            volume_1ES = load_nii_img[:, :, :, ES[patient]]
            volume_1ED_gt = load_nii_gt[:, :, :, ED[patient]]
            volume_1ES_gt = load_nii_gt[:, :, :, ES[patient]]

            ## Resizing
            reshaped_volume_1ED = []
            reshaped_volume_1ES = []
            reshaped_volume_1ED_gt = []
            reshaped_volume_1ES_gt = []

            for cut in range(volume_1ED.shape[2]):
                # Slices
                slice_ED = volume_1ED[:, :, cut]
                slice_ES = volume_1ES[:, :, cut]
                slice_ED_gt = volume_1ED_gt[:, :, cut]
                slice_ES_gt = volume_1ES_gt[:, :, cut]
                # To PIL
                slice_ED = Image.fromarray(volume_1ED[:, :, cut])
                slice_ES = Image.fromarray(volume_1ES[:, :, cut])
                slice_ED_gt = Image.fromarray(volume_1ED_gt[:, :, cut])
                slice_ES_gt = Image.fromarray(volume_1ES_gt[:, :, cut])
                # Resize
                resize = transforms.Resize(size=(384, 384))
                resize_mask = transforms.Resize(size=(384, 384), interpolation=Image.NEAREST)
                slice_ED = resize(slice_ED)
                slice_ES = resize(slice_ES)
                slice_ED_gt = resize_mask(slice_ED_gt)
                slice_ES_gt = resize_mask(slice_ES_gt)
                # To nparray
                slice_ED = np.array(slice_ED)
                slice_ES = np.array(slice_ES)
                slice_ED_gt = np.array(slice_ED_gt)
                slice_ES_gt = np.array(slice_ES_gt)
                # Normalise
                slice_ED *= 1.0/slice_ED.max()
                slice_ES *= 1.0/slice_ES.max()
                # Appending
                reshaped_volume_1ED.append(slice_ED)
                reshaped_volume_1ES.append(slice_ES)
                reshaped_volume_1ED_gt.append(slice_ED_gt)
                reshaped_volume_1ES_gt.append(slice_ES_gt)

                # pdb.set_trace()

                # Addition Slices
                if cut+1 == volume_1ED.shape[2]:
                    empty_slice = np.zeros((384,384))
                    distance = 16-volume_1ED.shape[2]
                    for i in range(distance):
                        reshaped_volume_1ED.append(empty_slice)
                        reshaped_volume_1ES.append(empty_slice)
                        reshaped_volume_1ED_gt.append(empty_slice)
                        reshaped_volume_1ES_gt.append(empty_slice)
            
            # Vertical stacking of slices
            volume_1ED_done = np.stack(reshaped_volume_1ED, axis=2)
            volume_1ES_done = np.stack(reshaped_volume_1ES, axis=2)
            volume_1ED_gt_done = np.stack(reshaped_volume_1ED_gt, axis=2)
            volume_1ES_gt_done = np.stack(reshaped_volume_1ES_gt, axis=2)

            print(volume_1ED_done.shape)
            print(volume_1ED_gt_done.shape)

            # np.savez('mmsdata/'+code[patient]+'_ED.npz', vol=volume_1ED_done, seg=volume_1ED_gt_done)
            # np.savez('mmsdata/'+code[patient]+'_ES.npz', vol=volume_1ES_done, seg=volume_1ES_gt_done)

            # np.savez('mmsdata/'+code[patient]+'_ED.npz', vol=volume_1ED_done)
            # np.savez('mmsdata/'+code[patient]+'_ES.npz', vol=volume_1ES_done)

            # # Loop for slices(cuts)
            # for cut in range(volume_1ED.shape[2]):
            #     slice_ED = volume_1ED[:, :, cut]
            #     slice_ES = volume_1ES[:, :, cut]
            #     slice_ED_gt = volume_1ED_gt[:, :, cut]
            #     slice_ES_gt = volume_1ES_gt[:, :, cut]

            #     # if patient==1 and cut==5: 
            #         # img = Image.fromarray(slice_ED)
            #         # img.show()
            #         # img = Image.fromarray(slice_ES)
            #         # img.show()
            #         # img = Image.fromarray(slice_ED_gt)
            #         # img.show()
            #         # img = Image.fromarray(slice_ES_gt)
            #         # print(img)
            #         # img.show()
            #         # print(np.max(slice_ED_gt))

            #     # Append ED and ES
            #     slices.append(tuple((slice_ED, slice_ED_gt)))
            #     slices.append(tuple((slice_ES, slice_ES_gt)))
    # return slices'mmsdata/'+
    return 

data_import(data_dir)

# volumes = data_import(data_dir)
# data = {'vol': volumes, 'seg': segmentations}
# np.savez('data.npz', volumes)

# data = data_import(data_dir)
# np.savez('data.npz', data)
# np.save('data.npz.npy', data)


def data_import_unlabeled(data_dir):
    # data directory
    # data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'
    unlabeled_data_dir = os.path.join(data_dir, 'Unlabeled/')

    slices = []

    for subdir in os.listdir(unlabeled_data_dir):
        if subdir != '.DS_Store':
            load_nii_img = nib.load(os.path.join(unlabeled_data_dir, subdir, subdir+'_sa.nii.gz'))
            load_nii_img = load_nii_img.get_fdata()

            for i in range(load_nii_img.shape[3]):
                volume = load_nii_img[:,:,:,i]

                for j in range(volume.shape[2]):
                    cut = volume[:,:,j]
                    slices.append(cut)
    return slices

# data_test = data_import_unlabeled(data_dir)
# np.save('data_test.npz.npy', data_test)


# data = np.load('data_test.npz.npy', allow_pickle=True)
# img = data[70]
# img = Image.fromarray(img)
# img.show()
# print(mask)
# mask = Image.fromarray((mask/3)*255)
# mask.show()