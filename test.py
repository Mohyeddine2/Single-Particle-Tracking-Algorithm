from lyso_tracker.lyso import Lyso
from lyso_tracker import descriptor, preprocess_data
import cv2

masque = cv2.imread("masks/01_C.tif",cv2.IMREAD_UNCHANGED)
binary = descriptor.binary_mask(masque, percentile=99)
all_detections,original_frames = preprocess_data.get_all_detections("test_live_cell.avi",binary,radius=4, percentile=95, sigma_log=15.0)

# Tracking
tracker = Lyso()
tracker.track(all_detections, dist_max=10, max_gap=10)
tracker.draw_tracks(original_frames, all_detections, masks=None, output_path="real data/cell_live.avi",fps=5)
tracker.save_csv("real data/cell_live.csv")
