import os
import json
import shutil
import tempfile
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


import pandas as pd
from functools import partial

import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import nnunetv2
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

import random
import pickle

from BraTS_2023_Metrics import metrics

def maybe_create_training_folder():

    # create folder for training in Dataset005
    
    if not os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr'):
        os.mkdir('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr')
    if not os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr'):
        os.mkdir('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr')
    if not os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/adjlabelsTr'):
        os.mkdir('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/adjlabelsTr')

def remove_training_folders():
    
    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr'):
        shutil.rmtree('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr')
    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr'):
        shutil.rmtree('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr')
    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/adjlabelsTr'):
        shutil.rmtree('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/adjlabelsTr')

def remove_preprocessing_data():

    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_preprocessed/Dataset005_Brats24T/'):
        shutil.rmtree('/home/riccardo/nnUNetv2/nnUNet_preprocessed/Dataset005_Brats24T/')

        

def send_file_to_train(list_subjects):

    for pt in list_subjects:

        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imagesTr/'+pt[:-7]+'_0000.nii.gz',
                   '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr/'+pt[:-7]+'_0000.nii.gz')
        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imagesTr/'+pt[:-7]+'_0001.nii.gz',
                   '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr/'+pt[:-7]+'_0001.nii.gz')
        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imagesTr/'+pt[:-7]+'_0002.nii.gz',
                   '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr/'+pt[:-7]+'_0002.nii.gz')
        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/imagesTr/'+pt[:-7]+'_0003.nii.gz',
                   '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/imagesTr/'+pt[:-7]+'_0003.nii.gz')
        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_raw/Dataset004_Brats24/labelsTr/'+pt,
                   '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr/'+pt)

def run_nnunet_step(step):

    extract_fingerprints([5], 'DatasetFingerprintExtractor', 22, True, True, True)
    plan_experiments([5], 'ResEncUNetPlanner' , 8, 'DefaultPreprocessor', None, None)
    with open('/home/riccardo/nnUNetv2/nnUNet_preprocessed/Dataset004_Brats24/nnUNetResEncUNetPlans.json') as json_data:
        d = json.load(json_data)
    d['dataset_name'] = 'Dataset005_Brats24T'
    with open('/home/riccardo/nnUNetv2/nnUNet_preprocessed/Dataset005_Brats24T/nnUNetResEncUNetPlans.json','w') as json_data:
        d = json.dump(d, json_data)
        
    preprocess([5], 'nnUNetResEncUNetPlans' , configurations= ['3d_fullres'], num_processes=[22])
    nnunet_trainer = get_trainer_from_args('005','3d_fullres', 1, 'nnUNetTrainer_100epochs_TopKloss', 'nnUNetResEncUNetPlans', None, device=torch.device('cuda'))
    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/checkpoint_final.pth')and step!=0:
        maybe_load_checkpoint(nnunet_trainer, None, None,'/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/checkpoint_final.pth')
    nnunet_trainer.run_training()
    if os.path.exists('/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/validation/'):
        shutil.rmtree('/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/validation/')
    nnunet_trainer.perform_actual_validation()
    


def analysis_validation_cases():

    summary_path = '/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/validation/summary.json'
    with open(summary_path) as f:
        d = json.load(f)

    results_for_pd = {}

    for i in d['metric_per_case']:
        sub = i['prediction_file'].split('/')[-1]
        results_for_pd[sub] = [i['metrics']['1']['Dice'],i['metrics']['2']['Dice'],i['metrics']['3']['Dice']]

    # results of validation cases
    sorted_results = pd.DataFrame(results_for_pd).T.mean(axis = 1).sort_values()

    # identify as "bad" cases with mean DICE lower than 0.8
    bad_sorted_results = pd.DataFrame(sorted_results[sorted_results<0.8])

    return bad_sorted_results

def analysis_validation_cases_challenge():

    validation_pred_path = '/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/validation/'
    validation_adjust_path = '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/adjlabelsTr/'
    original_seg = '/home/riccardo/nnUNetv2/nnUNet_raw/Dataset005_Brats24T/labelsTr/'
    # adjust_labels 2 --> 1
    # adjust_labels 1 --> 2

    for i in os.listdir(validation_pred_path):
        if not i.endswith('.nii.gz'):
            continue
        pre_mask = nib.load(validation_pred_path+i)
        pre_mask_ = np.array(pre_mask.dataobj)
        adj_pre_mask = np.zeros(pre_mask_.shape)
        adj_pre_mask[pre_mask_==2]=1
        adj_pre_mask[pre_mask_==1]=2
        adj_pre_mask[pre_mask_==3]=3
        new_image = nib.Nifti1Image(adj_pre_mask, affine=pre_mask.affine)
        nib.save(new_image,validation_adjust_path+i)
    results_for_pd = {}
    for i in os.listdir(validation_pred_path):
        if not i.endswith('.nii.gz'):
            continue
        r = metrics.get_LesionWiseResults(validation_adjust_path+i,original_seg+i,challenge_name='BraTS-GLI')
        results_for_pd[i] = [r['LesionWise_Score_Dice'].mean()]
    sorted_results = pd.DataFrame(results_for_pd).T.sort_values(by = 0)
    bad_sorted_results = pd.DataFrame(sorted_results[sorted_results[0]<0.8])

    return bad_sorted_results

def load_radiomics_dataset():

    dt_clusters = pd.read_excel('wt_radiomics_clusters_umap_dbscan.xlsx',index_col = 0)
    dt_clusters.index = [pt+'.nii.gz' for pt in dt_clusters.index.tolist()]

    return dt_clusters



def select_next_step_patients(bad_sorted_results,dt_clusters,already_seen_patients):

    bad_sorted_results['cluster'] = dt_clusters['cluster']

    # Select the cluster with worst predictions
    bad_cluster = bad_sorted_results.groupby('cluster').mean().sort_values(0).index[0]
    print('  Current bad cluster: ',bad_cluster)
    # Get patients from the worst cluster <=100
    list_patients_bad_clusters = dt_clusters[dt_clusters['cluster']==bad_cluster].index.tolist()
    bad_patients_next_step = random.sample(list_patients_bad_clusters,np.min([len(list_patients_bad_clusters),100]))

    # Get patients from a random other cluster <=100
    list_clusters = [-1,0,1,2,3,4,5,6,7]
    list_clusters.remove(bad_cluster)
    random_sampled_cluster = random.sample(list_clusters,1)[0]

    list_patients_other_cluster= dt_clusters[dt_clusters['cluster']==random_sampled_cluster].index.tolist()
    other_cluster_patients_next_step = random.sample(list_patients_other_cluster,np.min([len(list_patients_other_cluster),100]))
    print('  Random sampled cluster: ',random_sampled_cluster)
    # Sample subjects not already seen to reach 300 subjects

    n_random_to_300 = 300-len(bad_patients_next_step)-len(other_cluster_patients_next_step)

    print('  Patient sampling strategy: ',len(bad_patients_next_step),' bad cluster, ',len(other_cluster_patients_next_step),' other cluster and ',n_random_to_300,' random sampling')
    # Check to have at least 300 patients to be seen, otherwise clear the already_seen_patient list
    if len(already_seen_patients)+len(bad_patients_next_step)+len(other_cluster_patients_next_step)>=1900:
        already_seen_patients = []
    
    already_seen_patients.extend(bad_patients_next_step)
    already_seen_patients.extend(list_patients_other_cluster)
    already_seen_patients = list(set(already_seen_patients))

    to_sample_subjects = dt_clusters[~ dt_clusters.index.isin(already_seen_patients)].index.tolist()
    
    random_sampled_subjects = random.sample(to_sample_subjects,n_random_to_300)
    bad_patients_next_step.extend(other_cluster_patients_next_step)
    bad_patients_next_step.extend(random_sampled_subjects)

    return bad_patients_next_step,already_seen_patients



def main(step):
    # load excel with radiomics clusters

    dt_clusters = load_radiomics_dataset()
    
    if step>=1:

            bad_sorted_results = analysis_validation_cases_challenge()

            

    if not os.path.isfile('/home/riccardo/Documenti/Project/Politi/SegTumoriCerebrale/BraTS24/already_seen.pickle'):
    	    already_seen_patients = [] 
    	    print('Create list already seen')
    	    with open('/home/riccardo/Documenti/Projects/Politi/SegTumoriCerebrale/BraTS24/already_seen.pickle', 'wb') as f:
                pickle.dump(already_seen_patients, f)
            
    else:
            with open('/home/riccardo/Documenti/Project/Politi/SegTumoriCerebrale/BraTS24/already_seen.pickle', 'rb') as f:
                already_seen_patients = pickle.load(f)
                
                
    print('------ Iteration N°',step+1,'------\n\n')
    print('1- Check train folders')
    remove_training_folders()
    print('2- Remove preprocessing')
    remove_preprocessing_data()
    print('3- Create new training folders')
    maybe_create_training_folder()
    if step==0:
            patient_to_analyze = random.sample(dt_clusters.index.tolist(),300)
            print('5- Send files to train')
            send_file_to_train(patient_to_analyze)
            already_seen_patients.extend(patient_to_analyze)
            print('6- Run nnUNet')
            run_nnunet_step(step)
            with open('already_seen.pkl', 'wb') as f:
                pickle.dump(already_seen_patients, f)
    else:
            print('4 - Return patients to analyze')
            patient_to_analyze,already_seen_patients = select_next_step_patients(bad_sorted_results,dt_clusters,already_seen_patients)
            print('5- Send files to train')
            send_file_to_train(patient_to_analyze)
            already_seen_patients.extend(patient_to_analyze)
            already_seen_patients = list(set(already_seen_patients))
            with open('/home/riccardo/Documenti/Projects/Politi/SegTumoriCerebrale/BraTS24/already_seen.pickle', 'wb') as f:
                pickle.dump(already_seen_patients, f)
            
            print('6- Run nnUNet')
            run_nnunet_step(step)
    print('7- Run validation') 
    bad_sorted_results = analysis_validation_cases_challenge()
        
    if step>0:
        shutil.copy('/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/checkpoint_final.pth',
                   '/home/riccardo/nnUNetv2/nnUNet_results/checkpoint_final-'+str(step)+'pth')
    shutil.copy('/home/riccardo/nnUNetv2/nnUNet_results/Dataset005_Brats24T/nnUNetTrainer_100epochs_TopKloss__nnUNetResEncUNetPlans__3d_fullres/fold_1/checkpoint_final.pth',
                   '/home/riccardo/Dropbox/checkpoint_final-'+str(step)+'.pth')


if __name__ == "__main__":
    step = int(sys.argv[1])
    main(step)






    

        
