from smartsensor.smartsensor import processing_images, end2end_model

# Feature
indir = "examples/raw_data"
outdir = "examples/process_data"
processing_images(indir=indir, outdir=outdir)

# Model
train_rgb_path = "examples/process_data/ratio_normalized_roi/RGB_values.csv"
test_rgb_path = "examples/process_data/ratio_normalized_roi/RGB_values.csv"
train_concentration = "examples/raw_data/batch1.csv"
test_concentration = "examples/raw_data/batch2.csv"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
degree = 1
outdir = "examples/result"
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
metric
#        rmse       mae        r2   data
# 0  0.016956  0.014250  0.996320  train
# 0  0.045831  0.038247  0.973113   test
detail
#                 image expected_concentration predicted_concentration absolute_error     error   data
# 0   0.25-1_batch1.jpg                   0.25                0.267264       0.017264  0.017264  train
# 6  0.25-10_batch1.jpg                   0.25                0.241644       0.008356 -0.008356  train
# 4   0.25-8_batch1.jpg                   0.25                0.226544       0.023456 -0.023456  train
# 9   0.25-7_batch1.jpg                   0.25                0.274059       0.024059  0.024059  train
# 2   0.25-5_batch1.jpg                   0.25                0.243689       0.006311 -0.006311  train
