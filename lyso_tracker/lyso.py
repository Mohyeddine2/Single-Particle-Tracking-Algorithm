
from .tracker import nearest_neighbor_tracking
from .visualizer import draw_tracks_on_video, draw_detections_on_video
import csv
class Lyso:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
       # self.distance_max = distance_max
      #  self.max_gap = max_gap
        self.tracks = []

    def track(self, detections_by_frame, dist_max, max_gap):
        self.tracks = nearest_neighbor_tracking(
            detections_by_frame,
            distance_max=dist_max,
            max_gap=max_gap
        )

    def draw_tracks(self, frames, all_detections, masks, output_path, fps=10):
        draw_tracks_on_video(self.tracks, frames, all_detections, masks, output_path, fps)

    def draw_detections(self, frames, all_detections, masks, output_path, fps=10):
        draw_detections_on_video(frames, all_detections, masks, output_path, fps)



    def save_csv(self, path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "id", "x", "y"])
            for traj in self.tracks:
                for frame_index, track_id, (x, y) in traj:
                    writer.writerow([frame_index, track_id, x, y])

