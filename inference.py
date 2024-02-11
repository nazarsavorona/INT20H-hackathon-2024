import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary-weights",
        type=str,
        default="./weights/binary.pt",
        help="binary weights file",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default="./weights/detector.pt",
        help="detector weights file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./datasets/pneumonia_inference/images",
        help="source directory for images",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for inference",
    )
    return parser.parse_args()


def get_binary_predictions(binary_model, img_paths, batch_size):
    """
    Get binary predictions for a list of images

    Args:
    binary_model: YOLO model
    img_paths: list of image paths
    batch_size: batch size for inference

    Returns:
    df: dataframe with image ids and binary predictions
    """
    results = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]
        results.append(binary_model(batch))

    results = [item for sublist in results for item in sublist]

    probs = [result.probs.top1 for result in results]

    img_ids = [Path(x).stem for x in img_paths]

    df = pd.DataFrame({"patientId": img_ids, "pneumonia": probs, "path": img_paths})

    return df


def get_detector_predictions(detector_model, batch_size, binary_predictions):
    """
    Get detector predictions for a list of images

    Args:
    detector_model: YOLO model
    batch_size: batch size for inference
    binary_predictions: dataframe with binary predictions

    Returns:
    df: dataframe with image ids and detector predictions
    """

    detections = []
    old_size = 1024

    for i, row in binary_predictions.iterrows():
        img = row["path"]
        prediction_string = ""

        if row["pneumonia"] < 1:
            detections.append({'patientId': row['patientId'], 'PredictionString': prediction_string.strip()})
            continue

        results = detector_model(img)
        boxes = results[0].boxes
        if len(boxes.xywhn) > 1:
            for xywhn, confidence in zip(boxes.xywhn, boxes.conf):
                x, y, w, h = xywhn
                width = int(w * old_size) + 1
                height = int(h * old_size) + 1
                x_left = int(x * old_size - width / 2)
                y_top = int(y * old_size - height / 2)

                prediction_string += f"{confidence:.2f} {x_left} {y_top} {width} {height} "

        detections.append({'patientId': row['patientId'], 'PredictionString': prediction_string.strip()})

    return pd.DataFrame.from_dict(detections)


def main(args):
    batch_size = args.batch_size

    binary_weights = args.binary_weights
    detector_weights = args.detector_weights

    # Initialize
    binary_model = YOLO(binary_weights)

    # Get source images
    source = args.source
    files = [x for x in Path(source).glob("*")]

    # create dataframe with image paths
    img_paths = []
    for file in files:
        img_paths.append(str(file))

    df = get_binary_predictions(binary_model, img_paths, batch_size)

    detector_model = YOLO(detector_weights)

    # get detector predictions
    df = get_detector_predictions(detector_model, batch_size, df)

    # save results
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
