import nibabel as nib
import numpy as np
import pandas as pd
import os
import shutil
from torchvision import transforms
from PIL import Image

syn_path = '/home/carlesgc/Projects/nnUnetMIDL/mms/Synthetic/'
vols = '/home/carlesgc/Projects/nnUnetMIDL/mms/temp_split_raw/'
dir_img = '../midl2021/data_syn/OpenDataset/Training/Ready/'

example_out = 'imagesTr/A0S9V9_0000_A_1_0000.nii.gz' 
example_in_gt = 'Synthetic/A0S9V9_1_gt.nii.gz'
example_in = 'temp_split_raw/A0S9V9_0000.nii.gz'

df_path="/home/carlesgc/Projects/nnUnetMIDL/mms/201014_M&Ms_Dataset_Information_-_opendataset.csv"

# nii = nib.load('/home/carlesgc/Projects/nnUnetMIDL/mms/temp_split_gt/A0S9V9_0001.nii.gz')
# print(nii.affine)

# nii2 = nib.load('/home/carlesgc/Projects/nnUnetMIDL/mms/temp_split_raw/A0S9V9_0001.nii.gz')
# print(nii2.affine)

output_gt = '/home/carlesgc/Projects/nnUnetMIDL/nnUNet_raw/nnUNet_raw_data_base/nnUNet_raw_data/Task202_MNMsSyn/labelsTr/'
output_img = '/home/carlesgc/Projects/nnUnetMIDL/nnUNet_raw/nnUNet_raw_data_base/nnUNet_raw_data/Task202_MNMsSyn/imagesTr/'

table = pd.read_csv(df_path, index_col='External code')

def reshape(vol):
    reshaped_volume = []
    for cut in range(vol.shape[2]):
        # To PIL
        image = Image.fromarray(vol[:, :, cut])
        
        # Resize
        resize = transforms.Resize(size=(384, 384))
        
        image = resize(image)
        
        # To nparray
        image = np.array(image)
        
        # Normalise
        image *= 1.0/image.max()
        # Appending
        reshaped_volume.append(image)
        

        # Addition Slices
        if cut+1 == vol.shape[2]:
            empty_slice = np.zeros((384,384))
            distance = 16-vol.shape[2]
            for i in range(distance):
                reshaped_volume.append(empty_slice)
                    
        
    # Vertical stacking of slices
    volume_done = np.stack(reshaped_volume, axis=2)
    return volume_done

for (dirpath, dirnames, filenames) in os.walk(syn_path):
    for file in filenames:
        patient = file.split('_')[0]
        timeframe = file.split('_')[1]

        # print(patient)

        vendor = table.loc[patient, 'Vendor']
        centre = table.loc[patient, 'Centre']

        # print(table.loc[patient,'Vendor'])
        # print(file.strip('_gt.nii.gz'))
        
        #Loading
        gt = nib.load(syn_path+file)
        gt_data = gt.get_data()

        img = nib.load(vols+patient+'_'+'{0:04}'.format(int(timeframe))+'.nii.gz')
        img = img.get_data()

        #Reshape img
        img = reshape(img)

        #GT values
        gt_data = gt_data.astype(int).astype(float)

        name_img = output_img+patient+'_'+'{0:04}'.format(int(timeframe))+'_'+vendor+'_'+str(centre)+'_'+'0000'+'.nii.gz'
        name_gt = output_gt+patient+'_'+'{0:04}'.format(int(timeframe))+'_'+vendor+'_'+str(centre)+'.nii.gz'

        ## Save gt
        nii_gt = nib.Nifti1Image(gt_data, gt.affine, header=gt.header)
        nib.save(nii_gt, name_gt)
        # shutil.copy(os.path.join(syn_path,file), name_gt)

        # Save img
        nii_img = nib.Nifti1Image(img, gt.affine, header=gt.header)
        nib.save(nii_img, name_img)

        # print(nii_img.affine.shape)

# gt = nib.load('/home/carlesgc/Projects/nnUnetMIDL/mms/Synthetic/A0S9V9_1_gt.nii.gz')
# gt_data = gt.get_data()

# print(np.min(gt_data.astype(int).astype(float)))

# gt = nib.load('/home/carlesgc/Projects/nnUnetMIDL/mms/temp_split_gt/A0S9V9_0000.nii.gz')
# gt_data = gt.get_data()
# print(np.amax(gt_data))


# img = nib.load('/home/carlesgc/Projects/nnUnetMIDL/mms/temp_split_raw/A0S9V9_0001.nii.gz')
# vol = img.get_data()



    
# print(img.files)

# nii_img = nib.Nifti1Image(volume_done, gt.affine, header=gt.header)
# nib.save(nii_img, 'test.nii.gz')