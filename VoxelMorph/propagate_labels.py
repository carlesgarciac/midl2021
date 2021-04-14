import os
import numpy as np
from nibabel.testing import data_path
from scipy import ndimage
import pandas as pd
import nibabel as nib
import subprocess

# data_dir = '/Users/carlesgc/Projects/miccai-mms/data/Training-corrected/'
data_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/'
ready_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/Training/Ready/'
model_dir = '/home/carlesgc/Projects/midl2021/Voxelmorph/models/tfVolSeg3D.h5'
warp_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/Training/Warps/'
moved_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/Training/Moved/'


# synthetic_dir = '/home/carlesgc/Projects/midl2021/data_syn/OpenDataset/Training/Synthetic/'
synthetic_dir = '/home/carlesgc/Projects/nnUnetMIDL/mms/Synthetic/'


csv = pd.read_csv(os.path.join(data_dir, '201014_M&Ms_Dataset_Information_-_opendataset.csv'))
csv.set_index(csv.columns[0], inplace=True)
EDs = csv['ED']
ESs = csv['ES']


def register(patient, ED_volume, time_frame_volume):
    subprocess.run(['Voxelmorph/scripts/tf/register.py', '--moving', ready_dir+str(patient)+'_'+str(time_frame_volume)+'.npz', '--fixed', ready_dir+str(patient)+'_ED.npz', '--model', model_dir, '--moved', moved_dir+str(patient)+'_'+str(time_frame_volume)+'.npz','--warp', warp_dir+str(patient)+'_'+str(time_frame_volume)+'.npz'])
    # scripts/tf/register.py --moving mmsdata/A0S9V9_ES.npz --fixed mmsdata/A0S9V9_ED.npz --model models/tfVolSeg3D.h5 --moved playground/moved.nii.gz --warp playground/warp.nii.gz

def warp(patient, ED_gt, time_frame_volume):
    subprocess.run(['Voxelmorph/scripts/tf/warp.py', '--moving', ready_dir+str(patient)+'_ED_gt.npz', '--warp', warp_dir+str(patient)+'_'+str(time_frame_volume)+'.npz', '--moved', synthetic_dir+str(patient)+'_'+str(time_frame_volume)+'_gt.nii.gz'])
    # scripts/tf/warp.py  --moving playground/A0S9V9_ESgt.npz --warp playground/warp.nii.gz --moved playground/movedgt.nii.gz 

def propagate(data_dir):
    for subdirs, _, _ in os.walk(os.path.join(data_dir, 'Training/Labeled/')):
        patient = subdirs.split('/')[9]
        if patient != '':
            ES_frame = ESs[patient]
            ED_frame = EDs[patient]

            if ED_frame < ES_frame:
                for i in range(ED_frame, ES_frame+1):
                    register(patient, ED_frame, i)
                    warp(patient, ED_frame, i)

            if ES_frame < ED_frame:
                for i in range(ES_frame, ED_frame+1): 
                    register(patient, ED_frame, i)
                    warp(patient, ED_frame, i)


propagate(data_dir)