# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:30:39 2021

@author: Gani Perez
@GitHub: GaniPerez

This class will generate or load an array of MRI images to use for CNN processing.
"""

from MRI_Img import MRI
import os 
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import random

def test_model_stats(confusion_matrix):
    """using a confusion matrix, outputs accuracy, precision, sensitivity, and specificity.
    :param confusion_matrix: 2x2 confusion matrix
    :return: accuracy, precision, sensitivity, specificity"""
    TP = confusion_matrix[1,1]
    FP = confusion_matrix[0,1]
    TN = confusion_matrix[0,0]
    FN = confusion_matrix[1,0]
    
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN+FP)
    
    print(f'Accuracy =\t{accuracy}\nPrecision =\t{precision}\nSensitivity =\t{sensitivity}\nSpecificity =\t{specificity}')
    return accuracy, precision, sensitivity, specificity

def plot_images(images, cls_true, cls_pred=None, img_shape=(32,32,3)):
    """plots images with class value and predicted value (optionally)"""
    
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    

def plot_example_errors(data, cls_pred):
    """plots errors with plot_images"""
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    
    # reformats arrays into interpretable format
    cls_pred = cls_pred[0:]
    cls_true = np.argmax(data.test_y, axis=1)
    
    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_true)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test_x[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    
    # Get the true classes for those images.
    cls_true = cls_true[incorrect]
    
    # Plot the first 9 images.
    choices = random.choices(range(0,len(cls_pred)), k=9)
    plot_images(images=images[choices],
                cls_true=cls_true[choices],
                cls_pred=cls_pred[choices])

def one_hot_encode(cls_array, num_classes):
    """One-hot encodes classes by simple matrix multiplication.
    
    :param cls_values: 1D array of classes, coded by integer.
    :param num_classes: total number of classes to assess.
    :return one_hot: list of one-hot encoded classes."""
    
    if num_classes is None:
        num_classes = np.max(cls_array) + 1
    
    identity_matrix = np.eye(num_classes, dtype=float)
    
    return identity_matrix[cls_array]
    

class MRI_input_array:
    
    img_size = 32
    
    num_channel = 3
    
    img_size_flat = img_size * img_size * num_channel
    
    img_shape_full = (img_size, img_size, num_channel)
    
    num_classes = 2
    
    
    def __init__(self, input_dir=None, output_dir=None, load_dir=None, region='putamen', n=150, strides=3):
        """This will generate subsamples from MRIs if they do not exist, or load existing ones if they do exist.
        From there, it will one-hot encode the class as [control, patient] and split into a test-validation-train set.
        
        :param input_dir: directory where input images are, if subsamples are not yet generated. (default: None)
        :param output_dir: directory where subsamples should be output, if subsamples are not yet generated. (default: None)
        :param load_dir: directory where control and patient folders are located, if subsamples are generated. (default: None)
        :param region: region of interest -- putamen, hippocampus, NAc, pallidum, amygdala, thalamus, and caudate are supported
        :param n: number of subsamples to generate (Default: 150)
        :param strides: box distance between subsamples (Default 3)"""
        
        self.region = region
        self.region = region
        self.n = n
        self.strides = strides
        
        
        if not load_dir is None:
            
            print("Loading images from directory...")
            self.load_dir = load_dir
            
            img_vals_full, cls_list_full = self._load_subsamples(load_dir)
            
            # prints total number of images, as well as control vs patient
            print("imgs: " + str(len(img_vals_full)))
            print("num control: " + str(len([i for i in cls_list_full if i == 0])))
            print("num patient: " + str(len([i for i in cls_list_full if i == 1])))
            
            # sets these lists to arrays for matrix multiplication.
            img_arrays = np.asarray(img_vals_full)
            cls_arrays = np.asarray(cls_list_full)
            
            # Performs train-validation-test sample splitting
            self._train_test_split(img_array=img_arrays, cls_array=cls_arrays)
            
            # One-Hot Encodes classes
            self.train_y = one_hot_encode(self.train_y_cls, 2)
            self.validation_y = one_hot_encode(self.validation_y_cls, 2)
            self.test_y = one_hot_encode(self.test_y_cls, 2)
            
            print("train size: " + str(self.num_train))
            print("validation size: " + str(self.num_validation))
            print("test size: " + str(self.num_test))
            
            
            print("Finished loading images into object.")
            
        
        else:
            
            # assert (input_dir is None or output_dir is None), "You must specify an input and output directory as input_dir and output_dir."
            
            print("Generating new subsamples from input MRIs...")
            
            self.input_dir = input_dir
            self.output_dir = output_dir
            
            try:
                os.mkdir(output_dir)
                print("created output directory, continuing...")
            except:
                print("Output directory exists, continuing...")
            
            # Generates images using create_subsamples
            self._create_subsamples()
            
            self.load_dir = output_dir
            
            img_vals_full, cls_list_full = self._load_subsamples(self.load_dir)
            
            # prints total number of images, as well as control vs patient
            print("imgs: " + str(len(img_vals_full)))
            print("num control: " + str(len([i for i in cls_list_full if i == 0])))
            print("num patient: " + str(len([i for i in cls_list_full if i == 1])))
            
            # sets these lists to arrays 
            img_arrays = np.asarray(img_vals_full)
            cls_arrays = np.asarray(cls_list_full)
            
            self._train_test_split(img_array=img_arrays, cls_array=cls_arrays)
            
            # One-Hot Encodes classes
            self.train_y = one_hot_encode(self.train_y_cls, 2)
            self.validation_y = one_hot_encode(self.validation_y_cls, 2)
            self.test_y = one_hot_encode(self.test_y_cls, 2)
            
            print("train size: " + str(self.num_train))
            print("validation size: " + str(self.num_validation))
            print("test size: " + str(self.num_test))
            
            
            print("Finished loading images into object.")
            
            
            
    def _load_subsamples(self, load_dir):
        """Loads subsamples from directory, if they exist.
        :param load_dir: directory in which the control and patient folder are.
        :return: RGB_list and class_list, array of flat RGB images and integer class values."""
        
        control_dir = os.path.join(load_dir, "control")
        patient_dir = os.path.join(load_dir, "patient")
        

        control_sample_folders = [os.path.join(control_dir,f) for f in os.listdir(control_dir)
                                  if os.path.isdir(os.path.join(control_dir, f))]
        patient_sample_folders = [os.path.join(patient_dir,f) for f in os.listdir(patient_dir) 
                                  if os.path.isdir(os.path.join(patient_dir, f))]

        all_folders = control_sample_folders + patient_sample_folders

    
        RGB_list = []
        class_list = []
        
        for folder in all_folders:
            
            RGB_paths = [os.path.join(folder,i) for i in os.listdir(folder)
                         if i.endswith('.png')]
            
            for RGB_path in RGB_paths:
                
                RGBA = im.open(RGB_path)
                
                # Converting image from RGBA to RGB and changing to np array.
                RGB = RGBA.convert('RGB')
                RGB = np.asarray(RGB)
                
                # This renormalizes the values from 0 to 1.
                RGB_channels = []
                for i in range(0,3):
                    RGB_channel = RGB[:,:,i]
                    RGB_channel = RGB_channel / np.max(RGB_channel)
                    RGB_channels.append(RGB_channel)
                
                RGB_total = np.stack((RGB_channels[0],
                                     RGB_channels[1],
                                     RGB_channels[2]), axis = -1)
                
                # Reshapes image array into a flat array in order to load as input.
                RGB_flat = np.reshape(RGB_total, (self.img_size_flat))
                
                RGB_list.append(RGB_flat)
                if "control" in RGB_path:
                    class_list.append(0)
                elif "patient" in RGB_path:
                    class_list.append(1)
                else:
                    class_list.append(-1)
        
        return RGB_list, class_list
    
    
    def _create_subsamples(self, input_dir=None, output_dir=None):
        """Creates subsamples from input directory of MALPEM results.
        
        :param input_dir: location of all MALPEM results
        :param output_dir: location to save all outputs"""
        
        region = self.region
        
        if input_dir is None:
            input_dir = self.input_dir
        if output_dir is None:
            output_dir = self.output_dir
        
        sample_folders = [folder for folder in os.listdir(input_dir) if "sub" in folder]
        
        for folder in sample_folders:
            
            sample_path = os.path.join(input_dir,folder)
            MRI(input_dir=sample_path, output_dir=output_dir, region=region, n=self.n, strides=self.strides)
            
            
    def _train_test_split(self, img_array, cls_array, tvt_split = (0.8, 0.1, 0.1)):
        """Splits total images into train, validation, and test sets.
        
        :param img_array: array of objects (images) to split
        :param cls_array: matched array of classes
        :param tvt_split: tuple (length 3) signifying the train, validation, and test proportions.
        
        :return: None, sets train_x, train_y_cls, etc. as object values."""
        
        # divides 80% of the dataset into test group by randomly generating 80% of indices, then
        # setting the values of those indices into train_x and test_x.
        # Last, it will remove these values from the total list.
        
        
        print(img_array.shape[0])
        self.num_train = int(img_array.shape[0]*tvt_split[0])
        train_indices = sorted(np.random.choice(np.arange(img_array.shape[0]),
                                                size=self.num_train,
                                                replace=False), reverse=True)
        self.train_x, self.train_y_cls = img_array[train_indices], cls_array[train_indices]
        
        img_array = np.delete(img_array, obj=train_indices, axis=0)
        cls_array = np.delete(cls_array, obj=train_indices, axis=0)
            
        # Does the same for validation group: (10%)
        
        vt_split = tvt_split[1] / (tvt_split[1] + tvt_split[2])
        self.num_validation = int(vt_split * img_array.shape[0])
        validation_indices = sorted(np.random.choice(np.arange(img_array.shape[0]),
                                                     size=self.num_validation,
                                                     replace=False), reverse=True)
        self.validation_x, self.validation_y_cls = img_array[validation_indices], cls_array[validation_indices]
        
        img_array = np.delete(img_array, obj=validation_indices, axis=0)
        cls_array = np.delete(cls_array, obj=validation_indices, axis=0)
            
        # Finally, sets the remaining 10% to the test group.
        self.test_x, self.test_y_cls = img_array, cls_array
        self.num_test = self.test_y_cls.shape[0]
        
    def view_images(self, n=9, indices=None, img_list='test'):
        """Allows viewing of images from the test, train, or validation set.
        
        :param n: number of images to load
        :param indices: list of images to show
        :param img_list: which list to sample from (train, validation, or test)"""
        
        pass
        ### TODO ###
        