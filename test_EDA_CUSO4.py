from smartsensor.base import processing_images, end2end_model
import os


""" 
Dataset 1: Examples
Assignement:
1.Changing the inputs of end2end model, using the data from the 
raw_roi, delta_normalized_roi, ratio_normalized_roi. Firstly using batch 1
as training, batch 2 as testing. Draw the image and write simple obsevation.
2.Changing the inputs of train and test by take 2 over 3 datasets. Examples:
train=batch1,test=batch2,batch3; train=batch2,test=batch1,batch3;etc.
3.Draw figures to show the performance of the model.

"""
# Example
# Feature
data_path = "EDA/CUSO4"
indir = f"{data_path}/raw_data"
outdir = f"{data_path}/process_data"
processing_images(indir=indir, outdir=outdir)

# Model
train_rgb_path = f"{data_path}/process_data/ratio_normalized_roi/RGB_values.csv"
test_rgb_path = f"{data_path}/process_data/ratio_normalized_roi/RGB_values.csv"

train_rgb_path = f"{data_path}/process_data/delta_normalized_roi/RGB_values.csv"
test_rgb_path = f"{data_path}/process_data/delta_normalized_roi/RGB_values.csv"

train_concentration = f"{data_path}/raw_data/batch1.csv"
test_concentration = f"{data_path}/raw_data/batch2_and_3.csv"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
degree = 1
outdir = f"{data_path}/result"
prefix = "demo"

metric, detail = end2end_model(
    train_rgb_path,
    train_concentration,
    test_rgb_path,
    test_concentration,
    features,
    degree,
    outdir,
    prefix,
)
