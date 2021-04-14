import os
import subprocess
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import nibabel as nib

labels_path = '/home/carlesgc/Projects/nnUnetMIDL/nnUNet_raw/nnUNet_raw_data_base/nnUNet_raw_data/Task114_heart_MNMs/labelsTs/'

MNMs_model = '/home/carlesgc/Projects/nnUnetMIDL/MNMsoutputMNMs/'
MNMsSyn_model = '/home/carlesgc/Projects/nnUnetMIDL/MNMsSynoutputMNMs/'

labels_out = '/home/carlesgc/Projects/nnUnetMIDL/Results_scaners/Labels/'
MNMs_out =  '/home/carlesgc/Projects/nnUnetMIDL/Results_scaners/MNMs_model/'
MNMsSyn_out = '/home/carlesgc/Projects/nnUnetMIDL/Results_scaners/MNMsSyn_model/'

# for (dirpath, dirnames, filenames) in os.walk(labels_path):
#     for file in filenames:
#         scanner = file.split('_')[2].split('.')[0]
#         # print(file.split('_')[2].split('.')[0])
        
#         # print(str(os.path.join(labels_out,scanner,file)))
#         shutil.copy(os.path.join(labels_path,file), os.path.join(labels_out,scanner,file))

# for (dirpath, dirnames, filenames) in os.walk(MNMs_model):
#     for file in filenames:
#         if file != 'plans.pkl': 
#             if file !='summary.json':
#                 scanner = file.split('_')[2].split('.')[0]
#                 # print(file)
#                 # print(file.split('_')[2].split('.')[0])
                
#                 # print(str(os.path.join(labels_out,scanner,file)))
#                 shutil.copy(os.path.join(MNMs_model,file), os.path.join(MNMs_out,scanner,file))

# for (dirpath, dirnames, filenames) in os.walk(MNMsSyn_model):
#     for file in filenames:
#         if file != 'plans.pkl': 
#             if file !='summary.json':
#                 scanner = file.split('_')[2].split('.')[0]
#                 # print(file.split('_')[2].split('.')[0])
                
#                 # print(str(os.path.join(labels_out,scanner,file)))
#                 shutil.copy(os.path.join(MNMsSyn_model,file), os.path.join(MNMsSyn_out,scanner,file))


def convert_to_submission(source_dir, target_dir):
    niftis = subfiles(source_dir, join=False, suffix=".nii.gz")
    patientids = np.unique([i[:10] for i in niftis])
    maybe_mkdir_p(target_dir)
    for p in patientids:
        files_of_that_patient = subfiles(source_dir, prefix=p, suffix=".nii.gz", join=False)
        assert len(files_of_that_patient)
        files_of_that_patient.sort()
        # first is ED, second is ES
        shutil.copy(join(source_dir, files_of_that_patient[0]), join(target_dir, p + "_ED.nii.gz"))
        shutil.copy(join(source_dir, files_of_that_patient[1]), join(target_dir, p + "_ES.nii.gz"))

# convert_to_submission('/home/carlesgc/Projects/nnUnetMIDL/MNMSoutputACDC','/home/carlesgc/Projects/nnUnetMIDL/ACDC_submit_MNMs')
# convert_to_submission('/home/carlesgc/Projects/nnUnetMIDL/MNMsSynoutputACDC','/home/carlesgc/Projects/nnUnetMIDL/ACDC_submit_MNMsSyn')

dir_1 = '/home/carlesgc/Projects/nnUnetMIDL/ACDC_submit_MNMs/'
dir_2 = '/home/carlesgc/Projects/nnUnetMIDL/ACDC_submit_MNMsSyn/'

for (dirpath, dirnames, filenames) in os.walk(dir_1):
    for file in filenames:
        gt = nib.load(dir_1+file)
        gt_data = gt.get_data()

        gt_data[gt_data == 3] = 4
        gt_data[gt_data == 1] = 3
        gt_data[gt_data == 4] = 1
        # print(np.amax(gt_data))

        nii_gt = nib.Nifti1Image(gt_data, gt.affine, header=gt.header)
        nib.save(nii_gt, os.path.join(dir_1,file))