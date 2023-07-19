import cv2
import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os
import numpy as np
import statistics as st
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from typing import Any, List, Union, Tuple, Callable


def rgb2dataframe(array: Union[List[List[int]], ndarray]) -> DataFrame:
    """Convert RGB to dataframe. It is useful when we want to normalize the values.
    Notes: Using the normalized images causing the interval values

    Args:
        array (Union[List[List[int]], ndarray]): Just the array with 3 dimensions

    Returns:
        DataFrame: RGB dataframe
    """
    reshaped_array = array.reshape(-1, 3)
    column_names = ["R", "G", "B"]
    return pd.DataFrame(reshaped_array, columns=column_names)


def normalized_execute(
    roi_image: ndarray,
    background: ndarray,
    lum: Tuple[float, float, float],
    feature_method: Callable[[Any], float],
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Normalized the data according to the background as standard value

    Args:
        roi_image (ndarray): The array RGB of ROI
        background (ndarray): The array RGB of background
        lum (Tuple[float, float, float]): The standard value for normalization
        feature_method (Callable[[Any], float]): The mean or mode value to scale

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: The normalized of ratio, delta and raw RGB
    """

    # Take the upper region for balanced cover
    B = feature_method(background[:, :, 0])
    G = feature_method(background[:, :, 1])
    R = feature_method(background[:, :, 2])

    # Calculate the ratios for normalization
    ratioB = lum[0] / B
    ratioG = lum[1] / G
    ratioR = lum[2] / R

    # Normalize the ROI image using the ratios
    ratio_normalized_roi = rgb2dataframe(roi_image)
    ratio_normalized_roi["B"] *= ratioB
    ratio_normalized_roi["G"] *= ratioG
    ratio_normalized_roi["R"] *= ratioR

    # Calculate the deltas for adjustment
    deltaB = lum[0] - B
    deltaG = lum[1] - G
    deltaR = lum[2] - R

    # Adjust the ROI image using the deltas
    delta_normalized_roi = rgb2dataframe(roi_image)
    delta_normalized_roi["B"] += deltaB
    delta_normalized_roi["G"] += deltaG
    delta_normalized_roi["R"] += deltaR

    # Return the ratio-normalized, delta-adjusted, and original ROI images
    return ratio_normalized_roi, delta_normalized_roi, rgb2dataframe(roi_image)


def image_segmentations(
    indir: str,
    outdir: str,
    threshold: List = [(0, 110, 60), (80, 220, 160)],
    dim: List = [740, 740],
    bg_index: List = [50, 60, 350, 360],
    roi_index: int = 245,
) -> None:
    """Using the threshold to segment the images to different regions

    Args:
        indir (str): Images path
        outdir (str): The output path
        threshold (List): Cut-off values for segment
        dim (None): Dimesion
        bg_index (None): Background position for segment
        roi_index (None): ROI position for segment
    """

    # Create directories for storing intermediate files and results
    result_path = os.path.join(outdir, "result")
    os.makedirs(result_path, exist_ok=True)
    directories = [
        "squared_frame",
        "raw_roi",
        "ratio_normalized_roi",
        "delta_normalized_roi",
        "background",
    ]
    for directory in directories:
        os.makedirs(os.path.join(result_path, directory), exist_ok=True)

    # Contour value
    low_val = threshold[0]
    high_val = threshold[1]

    # Process each image in the input directory
    for image_location in glob.glob(os.path.join(indir, "*.jpg")):
        image = cv2.imread(image_location)
        file_name = os.path.splitext(os.path.basename(image_location))[0]

        # Create HSV image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image
        mask = cv2.inRange(hsv, low_val, high_val)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Select the largest contour
        largest_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > largest_area:
                cont = cnt
                largest_area = cv2.contourArea(cnt)

        # Get the parameters of the bounding box
        x, y, w, h = cv2.boundingRect(cont)
        squared_frame = image[y : y + h, x : x + w]  # noqa: E203

        # Section for cover
        roi = cv2.resize(squared_frame, dim, interpolation=cv2.INTER_AREA)

        # Background image
        background = roi[
            bg_index[0] : bg_index[1], bg_index[2] : bg_index[3]  # noqa: E203
        ]

        # ROI
        roi = roi[roi_index:-roi_index, roi_index:-roi_index]

        # File path
        squared_frame_path = os.path.join(
            result_path, "squared_frame", file_name + ".jpg"
        )
        brackground_path = os.path.join(result_path, "background", file_name + ".jpg")
        roi_path = os.path.join(result_path, "raw_roi", file_name + ".jpg")
        # Save
        cv2.imwrite(
            squared_frame_path,
            squared_frame,
        )
        cv2.imwrite(brackground_path, background)
        cv2.imwrite(roi_path, roi)


def balance_image(
    indir: str, outdir: str, constant: List, feature_method: Callable[[Any], float]
) -> None:
    """Image color balancing

    Args:
        indir (str): Images path
        outdir (str): Outdir path
        constant (List): constant for normalization
        feature_method (Callable[[Any], float]): methods for normalization [mean or mode]
    """

    result_path = os.path.join(outdir, "result")
    os.makedirs(result_path, exist_ok=True)

    for image_location in glob.glob(os.path.join(indir, "*.jpg")):
        print(f"Processing image: {image_location}")

        file_name = os.path.splitext(os.path.basename(image_location))[0]

        roi = cv2.imread(os.path.join(result_path, "raw_roi", file_name + ".jpg"))
        background = cv2.imread(
            os.path.join(result_path, "background", file_name + ".jpg")
        )

        # Normalize
        ratio_normalized, delta_normalized, roi = normalized_execute(
            roi_image=roi,
            background=background,
            lum=constant,
            feature_method=feature_method,
        )

        roi.to_csv(
            os.path.join(result_path, "raw_roi", file_name + ".csv"), index=False
        )
        ratio_normalized.to_csv(
            os.path.join(result_path, "ratio_normalized_roi", file_name + ".csv"),
            index=False,
        )
        delta_normalized.to_csv(
            os.path.join(result_path, "delta_normalized_roi", file_name + ".csv"),
            index=False,
        )


def get_rgb(indir: str, outdir: str, datatype: str) -> str:
    """Get the mean and mode value of R, G, B channel in the images in the
    directory. Then, save it to a dataframe

    Args:
        indir (str): Input directory
        outdir (str): Output directory

    Returns:
        DataFrame: Dataframe contains RGB of mean and mode value and relative
        image id
    """
    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)
    assert len(imgs_path) != 0, f"The directory {indir} does not contain any images"
    rgb_path = os.path.join(outdir, "RGB_values.csv")
    if os.path.exists(rgb_path):
        print(f"Skip! The raw RGB already features: \n {rgb_path}")
    else:
        with open(rgb_path, "w") as res:
            res.write("image,meanB,meanG,meanR,modeB,modeG,modeR\n")
            if datatype == "image":
                input_path = os.path.join(indir, "**.jpg")
                imgs_path = glob.glob(input_path)
                for img_path in imgs_path:
                    img_id = os.path.basename(img_path)
                    # Load image
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    b, g, r = cv2.split(img)
                    b_mode = st.mode(b.flatten())
                    g_mode = st.mode(g.flatten())
                    r_mode = st.mode(r.flatten())

                    res.write(
                        f"{img_id},{np.mean(b)},{np.mean(g)}, {np.mean(r)}, {b_mode}, {g_mode},{r_mode}\n"
                    )
            else:
                input_path = os.path.join(indir, "**.csv")
                imgs_path = glob.glob(input_path)
                for img_path in imgs_path:
                    img_id = os.path.basename(img_path)
                    # Load image
                    img = pd.read_csv(img_path)
                    b, g, r = img["B"], img["G"], img["R"]
                    b_mode = st.mode(b.flatten())
                    g_mode = st.mode(g.flatten())
                    r_mode = st.mode(r.flatten())

                    res.write(
                        f"{img_id},{np.mean(b)},{np.mean(g)}, {np.mean(r)}, {b_mode}, {g_mode},{r_mode}\n"
                    )
    return rgb_path


def get_data(rgb_path: str, concentration: str) -> DataFrame:
    """Combined the RGB features with their relative concentrations

    Args:
        rgb_path (str): The file path that contains the values for RGB features
        concentration (str): The file path that contains the values for concentration

    Returns:
        DataFrame: The combined dataframe that could be used for training and validating the
        machine learning models
    """
    df = pd.read_csv(rgb_path)
    conc = pd.read_csv(concentration)
    df = pd.merge(df, conc, on="image")
    return df


def train_regression(
    train: DataFrame, features: List, degree: int, outdir: str, prefix: str
) -> str:
    """Using the the training data for turning the regression model

    Args:
        train (DataFrame): dataframe with RGB values (features) and concentration (target)
        degree (int): polynomial degree
        outdir (str): the output directory
        prefix (str): the prefix name

    Returns:
        model: model path
    """
    # train
    x = train[features].values.astype(float)
    y = train["concentration"].values.astype(float)  # target variable
    poly = PolynomialFeatures(degree=degree)
    X_t = poly.fit_transform(x)
    clf = LinearRegression()
    clf.fit(X_t, y)

    # save models
    model_path = os.path.join(outdir, f"{prefix}_RGB_model.sav")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    return model_path


def evaluate_metrics(
    model: Any, data: DataFrame, features: List, degree: int
) -> (DataFrame, DataFrame):
    """Simple evaluation matrics for measure the errors

    Args:
        model (Any): the model object
        transform_model (Any): the transform data object
        x (np.array): the array of RGB for training the data
        y_real (np.array): the array of  concentration

    Returns:
        List: the score matrics and, the real and predicted value
    """
    # load
    x = data[features].values.astype(float)
    y_real = data["concentration"].values.astype(float)  # target variable
    reg_model = pickle.load(open(model, "rb"))
    poly = PolynomialFeatures(degree=degree)
    X_t = poly.fit_transform(x)
    y_pred = reg_model.predict(X_t)

    # metrics
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    mae = np.mean(np.abs(y_real - y_pred))
    r2 = r2_score(y_real, y_pred)
    metric = pd.DataFrame([[rmse, mae, r2]], columns=["rmse", "mae", "r2"])

    # result detail
    detail = pd.DataFrame([data["image"], y_real, y_pred])
    detail = detail.T
    detail.columns = ["image", "expected_concentration", "predicted_concentration"]
    detail = detail.sort_values("expected_concentration")
    detail["absolute_error"] = abs(
        detail["predicted_concentration"] - detail["expected_concentration"]
    )
    detail["error"] = (
        detail["predicted_concentration"] - detail["expected_concentration"]
    )

    return (metric, detail)


def processing_images(
    indir: str,
    outdir: str,
    threshold: List = [(0, 110, 60), (80, 220, 160)],
    dim: List = [740, 740],
    bg_index: List = [50, 60, 350, 360],
    roi_index: int = 245,
    constant: List = [60, 90, 30],
    feature_method: Callable[[Any], float] = np.mean,
) -> None:
    # Step 1: Extract ROI, background, and squared_frame return RGB value
    image_segmentations(
        indir=indir,
        outdir=outdir,
        threshold=threshold,
        dim=dim,
        bg_index=bg_index,
        roi_index=roi_index,
    )
    # Step 2: Balance images by normalizing using the background color and saving results
    balance_image(
        indir=indir,
        outdir=outdir,
        constant=constant,
        feature_method=feature_method,
    )


def end2end_model(
    train_rgb_path: str,
    train_concentration: str,
    test_rgb_path: str,
    test_concentration: str,
    features: str,
    degree: int,
    outdir: str,
    prefix: str,
):
    # Load data
    train_rgb = get_rgb(indir=train_rgb_path, outdir=outdir)
    train = get_data(rgb_path=train_rgb, concentration=train_concentration)
    test_rgb = get_rgb(indir=test_rgb_path, outdir=outdir)
    test = get_data(rgb_path=test_rgb, concentration=test_concentration)
    # Train
    features = features.split(",")
    train_model = train_regression(
        train=train, features=features, degree=degree, outdir=outdir, prefix=prefix
    )
    # Evaluate
    train_metric, train_detail = evaluate_metrics(
        model=train_model, data=train, features=features, degree=degree
    )

    test_metric, test_detail = evaluate_metrics(
        model=train_model, data=test, features=features, degree=degree
    )
    train_metric["data"] = "train"
    train_detail["data"] = "train"
    test_metric["data"] = "test"
    test_detail["data"] = "test"
    metric = pd.concat([train_metric, test_metric], axis=0)
    detail = pd.concat([train_detail, test_detail], axis=0)
    # Export data
    metric_path = os.path.join(outdir, "metric.csv")
    detail_path = os.path.join(outdir, "detail.csv")
    metric.to_csv(metric_path, index=False)
    detail.to_csv(detail_path, index=False)
    return metric, detail
