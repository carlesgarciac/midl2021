import os
import numpy as np
from nibabel.testing import data_path
from scipy import ndimage
import pandas as pd
import nibabel as nib
from torchvision import transforms
from PIL import Image
import pdb
import os

# data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'
data_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/'
csv = pd.read_csv(os.path.join(data_dir, '201014_M&Ms_Dataset_Information_-_opendataset.csv'))
csv.set_index(csv.columns[0], inplace=True)
EDs = csv['ED']
ESs = csv['ES']
# print(csv)
# print(ESs['A0S9V9'])


def get_it_ready(patient, ES_frame, ED_frame, load_nii_vols, load_nii_gt, time_frame):

    volume = load_nii_vols[:, :, :, time_frame]
    volume_gt = load_nii_gt[:, :, :, time_frame]


    reshaped_volume = []
    reshaped_volume_gt = []

    for cut in range(volume.shape[2]):
        # To PIL
        image = Image.fromarray(volume[:, :, cut])
        image_gt = Image.fromarray(volume_gt[:, :, cut])
        # Resize
        resize = transforms.Resize(size=(384, 384))
        resize_mask = transforms.Resize(size=(384, 384), interpolation=Image.NEAREST)
        image = resize(image)
        image_gt = resize_mask(image_gt)
        # To nparray
        image = np.array(image)
        image_gt = np.array(image_gt)
        # Normalise
        image *= 1.0/image.max()
        # Appending
        reshaped_volume.append(image)
        reshaped_volume_gt.append(image_gt)

        # Addition Slices
        if cut+1 == volume.shape[2]:
            empty_slice = np.zeros((384,384))
            distance = 16-volume.shape[2]
            for i in range(distance):
                reshaped_volume.append(empty_slice)
                reshaped_volume_gt.append(empty_slice)
    
    # Vertical stacking of slices
    volume_done = np.stack(reshaped_volume, axis=2)
    volume_gt_done = np.stack(reshaped_volume_gt, axis=2)

    # print(volume_done.shape)
    # print(volume_gt_done.shape)

    if time_frame == ES_frame:
        np.savez(data_dir+'/Training/Ready/'+str(patient)+'_ES.npz', vol=volume_done, seg=volume_gt_done)
        np.savez(data_dir+'/Training/Ready/'+str(patient)+'_ES_gt.npz', volume_gt_done)
    elif time_frame== ED_frame:
        np.savez(data_dir+'/Training/Ready/'+str(patient)+'_ED.npz', vol=volume_done, seg=volume_gt_done)
        np.savez(data_dir+'/Training/Ready/'+str(patient)+'_ED_gt.npz', volume_gt_done)
    else:
        np.savez(data_dir+'/Training/Ready/'+str(patient)+'_'+str(time_frame)+'.npz', volume_done)
    
    print(patient+'_at time frame '+str(time_frame)+' done')

def data_import(data_dir):
    # data directory
    # data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'

    for subdirs, _, _ in os.walk(os.path.join(data_dir, 'Training/Labeled/')):
            
            patient = subdirs.split('/')[9]
            
            if patient != '':
                ES_frame = ESs[patient]
                ED_frame = EDs[patient]
                
                load_nii_vols = nib.load(os.path.join(data_dir+ 'Training/Labeled/', patient+ '/',patient+'_sa.nii.gz'))
                load_nii_vols = load_nii_vols.get_fdata()
                load_nii_gt = nib.load(os.path.join(data_dir+ 'Training/Labeled/', patient+ '/',patient+'_sa_gt.nii.gz'))
                load_nii_gt = load_nii_gt.get_fdata()

                if ED_frame < ES_frame:
                    for i in range(ED_frame, ES_frame+1):
                        get_it_ready(patient, ES_frame, ED_frame, load_nii_vols, load_nii_gt, i)
                        # print(i)
                
                if ES_frame < ED_frame:
                    for i in range(ES_frame, ED_frame+1):
                        get_it_ready(patient, ES_frame, ED_frame, load_nii_vols, load_nii_gt, i)
                        # print(i)



            



    # labeled_data_dir = os.path.join(data_dir, 'Training/Labeled/')

    # # Excel sheet loading
    # xl_file = pd.read_csv(os.path.join(data_dir, '201014_M&Ms_Dataset_Information_-_opendataset.csv'))
    # dfs = {sheet_name: xl_file.parse(sheet_name)
    #         for sheet_name in xl_file.sheet_names}

    # code = dfs['TrainingAnonUnlabeled']['External code']  # External Code
    # ED = dfs['TrainingAnonUnlabeled']['ED']  # ED
    # ES = dfs['TrainingAnonUnlabeled']['ES']  # ES

    # slices = []


    # vols = []
    # segs = []

    # a, b, c = 0, 0, 0

    # for patient in range(len(dfs['TrainingAnonUnlabeled']['External code'])):
    #     # Vendor C is not segmented
    #     if dfs['TrainingAnonUnlabeled']['Vendor'][patient] != 'C':
    #         # Load and convert to np the nii files
    #         load_nii_img = nib.load(os.path.join(labeled_data_dir, code[patient], code[patient]+'_sa.nii.gz'))
    #         load_nii_img = load_nii_img.get_fdata()
    #         load_nii_gt = nib.load(os.path.join(labeled_data_dir, code[patient], code[patient]+'_sa_gt.nii.gz'))
    #         load_nii_gt = load_nii_gt.get_fdata()

    #         # Get time frame with Ground truth
    #         volume_1ED = load_nii_img[:, :, :, ED[patient]]
    #         volume_1ES = load_nii_img[:, :, :, ES[patient]]
    #         volume_1ED_gt = load_nii_gt[:, :, :, ED[patient]]
    #         volume_1ES_gt = load_nii_gt[:, :, :, ES[patient]]

    #         ## Resizing
    #         reshaped_volume_1ED = []
    #         reshaped_volume_1ES = []
    #         reshaped_volume_1ED_gt = []
    #         reshaped_volume_1ES_gt = []

    #         for cut in range(volume_1ED.shape[2]):
    #             # Slices
    #             slice_ED = volume_1ED[:, :, cut]
    #             slice_ES = volume_1ES[:, :, cut]
    #             slice_ED_gt = volume_1ED_gt[:, :, cut]
    #             slice_ES_gt = volume_1ES_gt[:, :, cut]
    #             # To PIL
    #             slice_ED = Image.fromarray(volume_1ED[:, :, cut])
    #             slice_ES = Image.fromarray(volume_1ES[:, :, cut])
    #             slice_ED_gt = Image.fromarray(volume_1ED_gt[:, :, cut])
    #             slice_ES_gt = Image.fromarray(volume_1ES_gt[:, :, cut])
    #             # Resize
    #             resize = transforms.Resize(size=(384, 384))
    #             resize_mask = transforms.Resize(size=(384, 384), interpolation=Image.NEAREST)
    #             slice_ED = resize(slice_ED)
    #             slice_ES = resize(slice_ES)
    #             slice_ED_gt = resize_mask(slice_ED_gt)
    #             slice_ES_gt = resize_mask(slice_ES_gt)
    #             # To nparray
    #             slice_ED = np.array(slice_ED)
    #             slice_ES = np.array(slice_ES)
    #             slice_ED_gt = np.array(slice_ED_gt)
    #             slice_ES_gt = np.array(slice_ES_gt)
    #             # Normalise
    #             slice_ED *= 1.0/slice_ED.max()
    #             slice_ES *= 1.0/slice_ES.max()
    #             # Appending
    #             reshaped_volume_1ED.append(slice_ED)
    #             reshaped_volume_1ES.append(slice_ES)
    #             reshaped_volume_1ED_gt.append(slice_ED_gt)
    #             reshaped_volume_1ES_gt.append(slice_ES_gt)

    #             # pdb.set_trace()

    #             # Addition Slices
    #             if cut+1 == volume_1ED.shape[2]:
    #                 empty_slice = np.zeros((384,384))
    #                 distance = 16-volume_1ED.shape[2]
    #                 for i in range(distance):
    #                     reshaped_volume_1ED.append(empty_slice)
    #                     reshaped_volume_1ES.append(empty_slice)
    #                     reshaped_volume_1ED_gt.append(empty_slice)
    #                     reshaped_volume_1ES_gt.append(empty_slice)
            
    #         # Vertical stacking of slices
    #         volume_1ED_done = np.stack(reshaped_volume_1ED, axis=2)
    #         volume_1ES_done = np.stack(reshaped_volume_1ES, axis=2)
    #         volume_1ED_gt_done = np.stack(reshaped_volume_1ED_gt, axis=2)
    #         volume_1ES_gt_done = np.stack(reshaped_volume_1ES_gt, axis=2)

    #         print(volume_1ED_done.shape)
    #         print(volume_1ED_gt_done.shape)

    #         np.savez('data_syn/OpenDataset/Training/Ready/'+code[patient]+'_ED.npz', vol=volume_1ED_done, seg=volume_1ED_gt_done)
    #         np.savez('data_syn/OpenDataset/Training/Ready/'+code[patient]+'_ES.npz', vol=volume_1ES_done, seg=volume_1ES_gt_done)

    #         # np.savez('mmsdata/'+code[patient]+'_ED.npz', vol=volume_1ED_done)
    #         # np.savez('mmsdata/'+code[patient]+'_ES.npz', vol=volume_1ES_done)

    #         # # Loop for slices(cuts)
    #         # for cut in range(volume_1ED.shape[2]):
    #         #     slice_ED = volume_1ED[:, :, cut]
    #         #     slice_ES = volume_1ES[:, :, cut]
    #         #     slice_ED_gt = volume_1ED_gt[:, :, cut]
    #         #     slice_ES_gt = volume_1ES_gt[:, :, cut]

    #         #     # if patient==1 and cut==5: 
    #         #         # img = Image.fromarray(slice_ED)
    #         #         # img.show()
    #         #         # img = Image.fromarray(slice_ES)
    #         #         # img.show()
    #         #         # img = Image.fromarray(slice_ED_gt)
    #         #         # img.show()
    #         #         # img = Image.fromarray(slice_ES_gt)
    #         #         # print(img)
    #         #         # img.show()
    #         #         # print(np.max(slice_ED_gt))

    #         #     # Append ED and ES
    #         #     slices.append(tuple((slice_ED, slice_ED_gt)))
    #         #     slices.append(tuple((slice_ES, slice_ES_gt)))
    # # return slices'mmsdata/'+
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