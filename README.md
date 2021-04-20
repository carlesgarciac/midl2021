# midl2021
This is the repository of the MIDL2021 submission for **"Semi-supervised Learning of Cardiac MRI using Image Registration"**

There are 3 important steps to replicate this work:
1. **Install nnUNet**: to install this, please follow the instructions of [nnUnet oficial repository](https://github.com/MIC-DKFZ/nnUNet)
2. **Install VoxelMorph**: to install this, please follow the instructions of [VoxelMorph oficial repository](https://github.com/voxelmorph/voxelmorph)
3. **Edit paths for scripts**: edit the paths in a way they match the data sources.

Once all this is ready the following commands and scripts should be called:
1. Run VoxelMorph/adapt.py. If your data is already in 3D you can skip this step.
2. Run VoxelMorph/data.py. If your 3D labelled data is already in a separate folder you can skip this step.
3. Run train.py [data folder] --model-dir [model folder]. This will train the VoxelMorph model used to propagate the labels
4. Run VoxelMorph/propagate_labels.py. This will result in synthetic labels for the unlabelled tim-frames
5. Run the scripts in nnUNet/dataset_conversion. This will get the data ready for training the nnUNet model. Run Task202_syn.py and Task202_NBNsSyn.py for getting the synthetic labeles ready.
6. In nnUNet tmp.py you will find some utils for solving posible naming errors between jobs, and preparing data for submission to ACDC (name and swap of channels for labels)