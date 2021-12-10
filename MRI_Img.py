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
import nibabel as nib
import random
from matplotlib.pyplot import imread, imsave
import time

class MRI:
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
    
    def __init__(self, input_dir=None, output_dir=None, load_dir=None):
        
        """Initializes the MRI_Img object. This object will contain one MRI image, plus stored characteristics such as
        path to the MRI image and mask. It will store each mask and MRI image as a np array, and also create a
        combined mask as a numpy array. It will automatically create sample images to store in the output_dir, and
        load data from output directory instead if needed."""
        
        self.subsample_size = 32
        
        if not load_dir is None:
            self._load_data(load_dir = load_dir)
        else:
            if input_dir is None:
                assert("You must pass a valid directory of the MALPEM output to input_dir.") 
            if output_dir is None:
                assert("You must specify the output directory to save subsamples in. output_dir")
            self.input_dir=input_dir
            self.output_dir=output_dir

            path_parts = [i for i in input_dir.split('/') if i != ""]
            sample_folder = path_parts[-1]
            self.sample_folder_no_date_time = sample_folder.split('_')[0]
            
            if "patient" in self.sample_folder_no_date_time:
                self.status = "patient"
            else:
                self.status = "control"

            # sets paths for the brain MRI image, as well as the putamen masks.
            self.path_brain = input_dir + "/" + self.sample_folder_no_date_time + "_T1w_N4_masked.nii.gz"
            self.path_putamen_left = input_dir + "/prob_MALPEM/posteriors_27.nii.gz"
            self.path_putamen_right = input_dir + "/prob_MALPEM/posteriors_28.nii.gz"

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
            self.RGB_subsamples, self.subsample_tuples = self._generate_subsamples()
            
            # save subsamples as images
            self._save_subsample_images(outdir = output_dir)
            
            # saves masks and total MRI as np array in master directory
            
            try:
                os.mkdir(os.path.join(output_dir, self.status, self.sample_folder_no_date_time,
                                      'master'))
            except:
                pass
            
            np.save(os.path.join(output_dir, self.status, self.sample_folder_no_date_time,
                                 'master', "brain.npy"),self.array_brain)
            np.save(os.path.join(output_dir, self.status, self.sample_folder_no_date_time,
                                 'master', "putamen_left.npy"), self.array_putamen_left)
            np.save(os.path.join(output_dir, self.status, self.sample_folder_no_date_time,
                                 'master', "putamen_right.npy"), self.array_putamen_right)
            np.save(os.path.join(output_dir, self.status, self.sample_folder_no_date_time,
                                 'master', "putamen_combined.npy"), self.array_putamen_combined)            
            
            
    def _generate_subsamples(self, n=150, prob_thresh=0.8, strides=3):
        """ generates n number of size*size RGB subsamples from the mask and original image.
        
        :param n: number of subsamples to generate
        :param prob_thresh: threshold probability to define a voxel as a 'hit' -- part of the region of interest
        :param strides: minimum box distance between subsample voxels
        :param size: dimensions of each subsample: must be even
        :return: list of RGB subsamples, and list of tuples they correspond to."""
        
        size = self.subsample_size
        
        tuples = self._generate_tuples_array(prob_thresh=prob_thresh)
        
        subsample_tuples = self._random_sample_tuples(tuples = tuples, n = n, strides = strides)
        
        RGB_subsamples = []
        
        for tup in subsample_tuples:
            RGB = self.extract_subsample_from_tuple(sample_tuple=tup, size=size) #### change
            RGB_subsamples.append(RGB)
        
        return RGB_subsamples, subsample_tuples
        
    def _generate_tuples_array(self, prob_thresh=0.8):
        
        """This takes a mask and generates tuples that correspond to areas of high-confidence in the mask.
        :param prob_thresh: threshold probability to define a voxel as a 'hit' -- part of the region of interest.
        :return: list of tuples; tuples are in list format."""
        
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
        comb_mask_array_tuples = np.array(np.nonzero(comb_mask_array_binary == 1))  # Finds elements
        tuples_array = np.transpose(comb_mask_array_tuples[:3,:])                   # This may be a 4D tensor, so we will
                                                                                    # cut it to 3D and transpose. It is in form
                                                                                    # row=tuples, columns=x,y,z of tuple
        tuples = tuples_array.tolist()
        return tuples

    def _random_sample_tuples(self, tuples, n, strides=3):
        """Sub-function for sampling n numbers of tuples which are at least "strides" manhattan units away from each other.
        Tuples should be in a list of lists.
        
        :param tuples: input tuples
        :param n: number of subsamples to create
        :param strides: min manhattan distance between tuples (default 3)
        :return: list of n tuples"""
        
        tuples = tuples
        
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
        
    def extract_subsample_from_tuple(self, sample_tuple, size):
        """Extracts a RGB image of size*size dimensions from a given tuple.
        
        :param sample_tuple: the tuple coordinates from which you would like to extract.
        :param size: size of subsample you would like to extract. This value MUST be even.
        :return: 3d array with dimensions (x, y, channel) -- RGB image"""
        
        x, y, z = sample_tuple[0], sample_tuple[1], sample_tuple[2]
        min_x, max_x, min_y, max_y, min_z, max_z = x-size/2, x+size/2, y-size/2, y+size/2, z-size/2, z+size/2
        
        min_x, max_x, min_y, max_y, min_z, max_z = int(min_x), int(max_x), int(min_y), int(max_y), int(min_z), int(max_z)
        
        pixel_array_z = self.array_brain[min_x:max_x,min_y:max_y,z]
        pixel_array_x = self.array_brain[x, min_y:max_y, min_z:max_z]
        pixel_array_y = self.array_brain[min_x:max_x, y, min_z:max_z]
        
        print("max pixel in x: " + str(np.max(pixel_array_x)))
        # normalizes values to range 0-1.
        x_norm = pixel_array_x / np.max(pixel_array_x)
        y_norm = pixel_array_y / np.max(pixel_array_y)
        z_norm = pixel_array_z / np.max(pixel_array_z)

        #### FIXME ####
        x_norm = x_norm[:,:,0]
        y_norm = y_norm[:,:,0]
        z_norm = z_norm[:,:,0]
        #### FIXME ####
        
        # Check to see the difference between x-z_norm on here vs the notebook.
        
        
        RGB_subsample = np.stack((x_norm, y_norm, z_norm), axis = -1)
        
        return RGB_subsample
        
    def _save_subsample_images(self, outdir):
        """Saves ndarray as tuple image using matplotlib.
        
        :param RGB: list of 3d arrays representing pixel values
        :param tup: list of tuples corresponding to RGB array
        :param outdir: location to save pictures
        """
        
        sample_name = self.sample_folder_no_date_time
        print("sample_name: " + sample_name)
        
        # makes necessary directories, if they don't exist
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        if not os.path.isdir(os.path.join(outdir, self.status)):
            os.mkdir(os.path.join(outdir, self.status))
        if not os.path.isdir(os.path.join(outdir, self.status, sample_name)):
            os.mkdir(os.path.join(outdir, self.status, sample_name))
        
        for i in range(0, len(self.RGB_subsamples)):
            RGB = self.RGB_subsamples[i]
            tup = self.subsample_tuples[i]
            
            tuple_str = [str(int) for int in tup]
            name_str = sample_name + "__" + "_".join(tuple_str) + ".png"
            
            path = os.path.join(outdir, self.status, sample_name,name_str)
            
            imsave(path, RGB, format='png')
            time.sleep(3)
        
        imsave(os.path.join(outdir, self.status, "dummyhelp.png"),x_array, format='png', cmap='gray')
            
            
    def _load_data(self,load_dir):
        """loads data from input directory. Load directory should be in form folder/[control-or-patient]/sample_name
        
        :param load_dir: directory of the sample that needs to be loaded."""
        
        subsample_pics = os.listdir(load_dir)
        subsample_pics = [i for i in subsample_pics if i.endswith(".png")]
        
        RGB_list = []
        tuple_list = []
        
        for pic in subsample_pics:
            pic_path = os.path.join(load_dir, pic)
            RGB = imread(pic_path)
            tup_string = subsample_pics[:-4].split("__")[1]
            tup = tup_string.split("_")
            RGB_list.append(RGB)
            tuple_list.append(tup)
        
        self.RGB_subsamples, self.subsample_tuples = RGB_list, tuple_list
        
        path_brain = os.path.join(load_dir,'master', 'brain.npy')
        path_putamen_left = os.path.join(load_dir, 'master', 'putamen_left.npy')
        path_putamen_right = os.path.join(load_dir, 'master', 'putamen_right.npy')
        path_putamen_combined = os.path.join(load_dir, 'master', 'putamen_combined.npy')
        
        self.array_brain = imread(path_brain)
        self.array_putamen_left = imread(path_putamen_left)
        self.array_putamen_right = imread(path_putamen_right)
        self.array_putamen_combined = imread(path_putamen_combined)
        
        self.status = load_dir.split("/")[-2]
        
                                                   
                                                   
                                                   
        
        