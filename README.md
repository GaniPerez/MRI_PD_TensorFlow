# MRI_PD_TensorFlow

## Introduction
This repo consists of python code and data files needed to train a convolutional neural network to distinguish between 3T T1-weighted MRIs of Parkinson's Disease patients versus relatively age-matched, healthy controls. The cohort consists of 3T T1-weighted MRIs of 36 healthy patients and 47 PD cases from the Neuro-consortium (neurocon) and Tao Wu dataset (taowu). Downloads for both datasets is available [here](http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html).

*Acknowledgement:* If you use re-adapt this workflow for any other dataset, please be sure to cite:
**Brain Extraction**
C. Ledig, R. A. Heckemann, A. Hammers, J. C. Lopez, V. F. J. Newcombe, A. Makropoulos, J. Loetjoenen, D. Menon and D. Rueckert, "Robust whole-brain segmentation: Application to traumatic brain injury", Medical Image Analysis, 21(1), pp. 40-58, 2015.
R. Heckemann, C. Ledig, K. R. Gray, P. Aljabar, D. Rueckert, J. V. Hajnal, and A. Hammers, "Brain extraction using label propagation and group agreement: pincram", PLoS ONE, 10(7), pp. e0129211, 2015.
**Subsample Generation**
W. Lin, T. Tong, Q Gao., D. Guo, Y. Yang, G. Guo, M. Xiao, M. Du, X. Qu, and The Alzheimer's Disease Neuroimaging Initiative, "Convolutional neural networks-based MRI image analysis for the Alzheimer's disease prediction from mild cognitive impairment", Frontiers in Neuroscience, 2018.

### The Workflow
The workflow is as follows:
1. Masks for various brain regions were extracted from 3T T1-weighted MRIs using MALPEM/pincram software.
2. Voxel coordinates of the regions of interest were extracted from masks, and 50 voxel coordinates were sampled from the total list of voxel coordinates such that no voxel coordinate was within a 5x5x5 voxel block of another.
3. For each sampled coordinate, an axial, coronal, and sagittal cut (32x32) was extracted and overlayed into a single RGB image
4. Subsamples were split into train-test-validation sets (0.8, 0.1, 0.1) and run through the CNN for 41 epochs to train.

## Installation
The pipeline was created on WSL2 (Windows Subsystem for Linux) using Ubuntu 20.04. However, the individual constituents of the workflow should also be compatible with any linux distribution or virtual machine. Unfortunately, MALPEM/pincram is not supported on Mac OS X, but the outputs of MALPEM will also be provided in a tar.gz file to test the downstream workflow.

### System Requirements
**OS and Machine Specs:**
- Linux is highly recommended, though this workflow can also be performed through WSL on Windows 10.
- MALPEM and tensorflow should run on any 64-bit machine. Furthermore, the network architecture is not deep; the neural network should run on a CPU allocated with 16GB RAM, or a GPU with 8GB VRAM. However, a GPU is strongly recommended for the hyperparameter tuning process.

**Memory Allocation**
- It is suggested to have at least 16GB of RAM available (to your virtual machine, if applicable). More will be necessary if running the neural network from a CPU.

**Disk Space**
- These file sizes are quite large. The MALPEM installation requires about 10GB of disk space, and the sum total of output files from this project can run up to 50GB. Beware.

### Prerequisites
1. Download and install MALPEM [here](https://github.com/ledigchr/MALPEM). 
2. Download conda (or another similar platform) and install nibabel, nilearn, matplotlib, numpy, and tensorflow-gpu. This workflow was developed with tensorflow-nightly (2.8.0-dev) due to GPU restrictions, but it should hypothetically work with any tensorflow build >2.3.0.
3. Clone this git repository.

## Running the Workflow
Running the workflow should be relatively straightforward. The general directions are documented here, but most directions for building the network architecture and training the model are also found in the jupyter notebook file, MRI_PD_Tensorflow.ipynb.

1. Download the datasets found [here](http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html). Conglomerate all patient and control directories into a single folder called "./input_MRIs" located at the root directory of the git clone directory. The structure should look like this:
  - input_MRIs
    - sub-control_...
    - sub-control_...
    - ...
    - sub_patient_...
    - sub_patient_...
2. Go to the git clone directory on your terminal and run malpem_proot_bulk.sh with ```sh malpem_proot_bulk.sh```. MALPEM will normalize each MRI image, put it into voxel space, generate masks for the skull to remove it, then segment the brain into various brain regions of interest. This will take a few days to run -- at t=8 threads, each MRI takes about 40min-1hr to process. Alternatively, you may ask me for the MALPEM-processed output masks.
3. Open up the jupyter notebook and run through all cells. The notebook is written tutorial-style with markdown explaining most steps, so it should be fairly easy to follow along.
