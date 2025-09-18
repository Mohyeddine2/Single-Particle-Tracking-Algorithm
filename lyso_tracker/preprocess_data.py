from lyso_tracker.lyso import Lyso
from lyso_tracker import descriptor
import cv2
import os
import natsort
from tifffile import imread
from tqdm import tqdm

import numpy as np
from pathlib import Path
def normalize(image_path):
    image = imread(image_path)
    image_d = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Erreur : Impossible de lire {image_path}")
        return None
    normalized = cv2.normalize(image_d, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype("uint8")
def get_all_detections(path,radius, percentile, sigma_log):
    images = get_data(path)
    partical_data =  []
    for frame in tqdm(images):  # Limiter à 100 images pour la démo
        partical_data.append(descriptor.detect_particles_sef(frame, radius, percentile, sigma_log))
    return partical_data, images
def get_data(path):
    frames = []
    if path.endswith((".mp4", ".avi", ".mov")):
        video = cv2.VideoCapture(path)
        while True:
            ret, frame = video.read()
            normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

            if not ret:
                break
            frames.append(normalized)
        video.release()
    else:
        images = natsort.natsorted([img for img in os.listdir(path) if img.endswith((".tiff", "TIF", ".jpeg", ".png"))])
        for img_name in images:  # Limiter à 100 images pour la démo
            img_path = os.path.join(path, img_name)
            frame = normalize(img_path)
            frames.append(frame)
    return frames



