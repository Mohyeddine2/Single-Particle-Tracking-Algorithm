from multihypo.mht import MHT
from multihypo.plot_tracks import plot_2d_tracks # ton fichier contenant la fonction
import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
import cv2
import math
import os
import natsort

image_folder = 'LysoTracker Expt A\LysoTracker Expt A'

images = [img for img in os.listdir(image_folder) if img.endswith((".TIF", ".jpeg", ".png"))]
sorted_images = natsort.natsorted(images)

def normalize(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire l'image en niveaux de gris
    if image is None:
        print(f"Erreur : Impossible de lire {image_path}")
        return None
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype("uint8")
def detect_particles(image, radius=2, percentile=90):
    threshold = np.percentile(image, percentile)
    local_max = (image == maximum_filter(image, size=2 * radius + 1))
    local_max = local_max & (image > threshold)  #
    labeled, num_features = label(local_max)

    # Compute intensity-weighted centroid for each detected particle
    centroids = center_of_mass(image, labeled, range(1, num_features + 1))
    centroids = np.array(centroids)  # Convert to NumPy array
    return centroids


partical_data = []
for i, image_name in enumerate(sorted_images):
    image_path = os.path.join(image_folder, image_name)
    frame = normalize(image_path)
    detected_positions = detect_particles(frame, radius=2, percentile=95)
    image_with_particles = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Ajouter les points détectés en rouge
    partical_data.append(detected_positions)

detections = partical_data[:10]
params = {
    "v": 640 * 480,
    "dth": 1000,
    "k": 0,
    "q": 1e-5,
    "r": 0.01,
    "n": 1,
    "bth": 10,
    "nmiss": 1,
    "pd": 0.9
}

mht_tracker = MHT(detections, params)
trajectories = mht_tracker.run()

# Sauvegarde les trajectoires dans un fichier CSV
import csv

output_csv = "trajectoires.csv"
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track', 'u', 'v'])  # header
    for track_id, track in enumerate(trajectories):
        for frame_idx, coords in enumerate(track):
            if coords is not None:
                writer.writerow([frame_idx, track_id, coords[0], coords[1]])
            else:
                writer.writerow([frame_idx, track_id, 'None', 'None'])

# Affiche les trajectoires à partir du fichier généré
plot_2d_tracks(output_csv, Flag_Save=False)

