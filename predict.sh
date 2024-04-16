#!/bin/bash


mkdir /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/labelspredv2_1
nnUNetv2_predict -i /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imgVal/ -o /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/labelspredv2_1 -d 5 -p nnUNetResEncUNetPlans -c 3d_fullres -tr nnUNetTrainer_100epochs_TopKloss -f 1 -chk /home/riccardo/nnUNetv2/nnUNet_results/checkpoint_final-1.pth 

mkdir /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/labelspredv2_2
nnUNetv2_predict -i /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imgVal/ -o /home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/labelspredv2_2 -d 5 -p nnUNetResEncUNetPlans -c 3d_fullres -tr nnUNetTrainer_100epochs_TopKloss -f 1 -chk /home/riccardo/nnUNetv2/nnUNet_results/checkpoint_final-2.pth 


