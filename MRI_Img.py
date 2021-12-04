#############################################################

# By: Gani Perez
# Git: GaniPerez
# 12/10/2021

# Converts 3T T1-MRI images from PD datasets into usable 32x32x32 images subsamples from the putamen.

# The datasets can be found and downloaded from the link: http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html

# The dataset must then be processed via MALPEM in order to generate whole brain masks (to remove the skull), as wellas a probability distribution (labeled "posterior_x.nii.gz") that is a mask, but we will instead use
# it to generate the voxel coordinates for the left and right putamen. 

#############################################################

import os
import numpy as np
import pandas as pd
from dataset import one_hot_encoded
from nilearn import image, regions, plotting, input_data, masking
import nibabel as nib
import random

class MRI_Img:
    ''' This class takes an input of MRI image paths and uses these pictures for display, to use as a
    template for extracting 3 32x32 "subsamples" for machine learning processing. For each voxel coord
    corresponding to a location in the target tissue of interest, the class will generate a coronal,
    sagittal, and axial 32x32 cut.'''
    
    # our default dimensions of each subsample (three slices of 32x32 each)
    subsample_size = 32
    
    # number of channels is 3 due to 3 different cuts. They will be overlayed in RGB format.
    num_channels = 3
    
    # the size of the flattened image array. Three channels, 32x32 pixels each.
    subsample_size_flat = subsample_size * subsample_size * num_channels
    
    # number of classes: in our case, PD versus control.
    num_classes = 2
    
    def __init__(self, data_dir=None, output_dir=None):
        
        """Initializes the MRI_Img object. This object will contain one MRI image, plus stored characteristics such as
        path to the MRI image and mask. It will store each mask and MRI image as a np array, and also create a
        combined mask as a numpy array. It will automatically create sample images to store in the output_dir, and
        load data from output directory instead if needed."""
        
        
        if data_dir is None:
            assert("You must pass a valid directory of the MALPEM output to data_dir.") 
        if output_dir is None:
            assert("You must specify the output directory to save subsamples in. output_dir")
        elif os.path.isdir(output_dir):
            ### LOAD DATA ###
        else:
            self.data_dir=data_dir
            self.output_dir=output_dir

            path_parts = [i for i in data_dir.split('/') if i != ""]
            sample_folder = path_parts[-1]
            sample_folder_no_date_time = sample_folder.split('_')[0]

            # sets paths for the brain MRI image, as well as the putamen masks.
            self.path_brain = data_dir + "/" + sample_folder_no_date_time + "_T1w_N4_masked.nii.gz"
            self.path_putamen_left = data_dir + "/prob_MALPEM/posteriors_27.nii.gz"
            self.path_putamen_right = data_perez + "/prob_MALPEM/posteriors_28.nii.gz"

            # loads images (temporarily) to extract voxel information.
            MRI_brain = nib.load(self.path_brain)
            MRI_putamen_left = nib.load(self.path_putamen_left)
            MRI_putamen_right = nib.load(self.path_putamen_right)

            # load image and keep the ndarray of pixels for visualization and processing
            self.array_brain = MRI_brain.get_fdata()
            self.array_putamen_left = MRI_putamen_left.get_fdata()
            self.array_putamen_right = MRI_putamen_right.get_fdata()

            # combine masks to form a single mask consisting of both putamen
            self.array_putamen_combined = np.maximum(self.array_putamen_left, self.array_putamen_right)

            # apply mask to generate subsamples
            ### generate_subsamples()
            
            # save subsamples as images
            ### save_img()
        
    def _generate_subsamples(self, n=150, prob_thresh=0.8):
        
        """This takes the mask and generates 150 putamen subsamples that are at least 3 manhattan units away from each other
        (in voxels). Each putamen subsample will 3 slices (coronal, sagittal, and axial) of the putamen of size 32x32
        pixels.
        
        :param n: number of subsamples (default 150)
        :param prob_thresh: probability threshold to define a voxel as a 'hit' -- part of the region of interest
        :return: array of 32x32x3 representing image cuts"""
        
        # I'm not really sure whether this assignment creates a copy, so I will not modify comb_mask_array and copy this
        # value instead. comb_mask_array stands for combined mask array.
        comb_mask_array = self.array_putamen_combined
        comb_mask_array_binary = comb_mask_array.copy()
        
        # comb_mask_array (and masks in general) create a probability density suggesting how confident MALPEM
        # is that a given voxel is of the specified brain region (putamen). We will turn this probability density
        # into a binary filter to more easily apply it. We do this by setting a probability threshold: anything
        # above a given probability is considered truly part of the putamen, and anything below is not.
        
        comb_mask_array_binary[comb_mask_array_binary >= 0.8] = 1             # reassigning hits to 1
        comb_mask_array_binary[comb_mask_array_binary < 0.8] = 0              # reassigning misses to 0
        
        # We must generate n number of 3D tuples that are within the mask, then set the tuples as an attribute of the 
        # object.
        comb_mask_array_tuples = np.array(np.nonzero(mask_array2 == 1))             # Finds elements
        tuples_array = np.transpose(comb_mask_array_tuples[:3,:])                   # This may be a 4D tensor, so we will
                                                                                    # cut it to 3D.
        tuples = tupes_array.tolist()
        
        self.subsample_tuples = tuples
        
        _random_sample_tuples(self.subsample_tuples, n = 150, strides = 3)
        
        # Now to randomly sample 150 tuples. I will brute-force this, but there must be a more efficient, elegant, and
        # more pythonic way to do this!
        
    def _random_sample_tuples(tuples, n, strides=3):
        """Sub-function for sampling n numbers of tuples which are "strides" manhattan units away from each other.
        Tuples should be in a list of lists.
        
        :param tuples: list of tuples
        :param n: number of subsamples to create
        :param strides: min manhattan distance between tuples (default 3)
        :return: list of n tuples"""
        
        # generates 500 random tuples from the total list of tuples.
        rand_tuples = random.sample(range(0,len(tuples)), 500)
        
        chosen_tuples = []
        generate_bad_tuples = []
        
        # checks each tuple to see if tuple is compatible with the previous ones.
        for i in rand_tuples:
            new_tuple = tuples[i]
            if new_tuple in generate_bad_tuples:
                continue                                                        # continues if tuple is in list of
                                                                                # disallowed tuples
            x = new_tuple[0]
            y = new_tuple[1]
            z = new_tuple[2]
            for a,b,c in zip(range(x-strides,x+strides),range(y-strides,y+strides),range(z-strides,z+strides)):
                generate_bad_tuples.append([a,b,c])                             # populates list of disallowed tuples
                                                                                # when adding a new good tuple to the list
    
            chosen_tuples.append(new_tuple)                                     # If it reaches here, it is a good tuple.
            if len(chosen_tuples) >= n:
                break                                                           # finishes when n tuples have been found.

        return chosen_tuples
        
        
        
        
        
        
        
        