# EPFL_Machine_Learning_Project1

## Higgs Boson Classification by Machine Learning

First project for Machine Learning course (CS-443) at EPFL. It gives several machine learning algorithms realized by numpy to find the Higgs boson using original data from CERN. The report explains the details of data cleaning, feature engineering and machine learning procedure, and the python scripts are in the root folder.

This algorithm obtained a classification accuracy of 0.82 on AIcrowd platform.

* [Getting started](#getting-started)
    * [Data](#data)
    * [Folders and Files](#folders-and-files)
* [Running the code](#running-the-code)
* [Contact us](#contact-us)

## Getting started

### Data

The raw data can be downloaded on the AIcrowd platform: [AIcrowd | EPFL Machine Learning Higgs | Challenges](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs). Note that the original data downloaded must remain unchanged and directly loaded. Moreover, for loading the data properly, the users should change the data paths in the `run.py` file to their own paths. The file for test submission will show in the root folder once the python script finishes running.


### Folders and Files

`run.py` is the main script, 

`implementations.py` contains core machine learning algorithms,  

`preprocessing.py` contains codes for raw data cleaning and feature engineering, 

`helpers.py` provides some help functions for figure plotting, data reading and saving, data standardization and cross validation, etc.,

`report.pdf`: project report explaining our machine learning procedure in .pdf format written in Latex,

`project1_description.pdf`: assignment description given by EPFL,

`requirements`: environment used in the project.

## Running the code

Move to the root folder and execute:

    python run.py

Detailed parameters for tuning the algorithm is well documented in `report.pdf`.

## Contact us
Please don't hesitate to contact the authors about any questions about the project, data or algorithms in general:

* Tianyu Gu: tianyu.gu@epfl.ch
* Qianqing Wang: qianqing.wang@epfl.ch
* Xinyu Liu: xinyu.liu@epfl.ch

@ 2022 Tianyu Gu