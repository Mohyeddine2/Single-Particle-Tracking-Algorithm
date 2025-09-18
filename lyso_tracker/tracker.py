
import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbor_tracking(detections_by_frame, distance_max, max_gap):
    """
Particle tracking using nearest neighbor + gap closing.

Parameters:
    detections_by_frame: list of (x, y) arrays per frame
    distance_max: maximum distance to associate points
    max_gap: maximum number of missing frames before closing a track

Returns:
    trajectories: list of trajectories, each
    trajectory is a list of (frame_index, track_id, (x, y))
"""

    trajectories = []
    active_tracks = []
    next_track_id = 0

    for frame_index, detections in enumerate(detections_by_frame):
        if len(detections) > 0:
            det_points = np.array(detections)
            if det_points.ndim == 1:
                det_points = det_points.reshape(1, -1)
            elif det_points.ndim != 2 or det_points.shape[1] != 2:
                print(f"Frame {frame_index} contient des données invalides : {det_points}")
                continue
            tree = cKDTree(det_points)
        else:
            tree = None

        matched_tracks = set()
        matched_detections = set()

        if tree is not None:
            for track in active_tracks:
                dist, det_idx = tree.query(track['last_pos'], distance_upper_bound=distance_max)
                if dist != np.inf and det_idx < len(det_points) and det_idx not in matched_detections:
                    pos = tuple(det_points[det_idx])
                    trajectories[track['id']].append((frame_index, track['id'], pos))
                    track['last_pos'] = pos
                    track['last_frame'] = frame_index
                    track['miss_count'] = 0
                    matched_tracks.add(track['id'])
                    matched_detections.add(det_idx)
                else:
                    track['miss_count'] += 1
        else:
            for track in active_tracks:
                track['miss_count'] += 1

        if tree is not None:
            for i, pt in enumerate(det_points):
                if i not in matched_detections:
                    track_id = next_track_id
                    next_track_id += 1
                    trajectories.append([(frame_index, track_id, tuple(pt))])
                    active_tracks.append({
                        'id': track_id,
                        'last_pos': tuple(pt),
                        'last_frame': frame_index,
                        'miss_count': 0
                    })

        active_tracks = [trk for trk in active_tracks if trk['miss_count'] <= max_gap]

    longueur_min = 0
    trajectoires_filtrées = [traj for traj in trajectories if len(traj) >= longueur_min]

    return trajectoires_filtrées
