
"""
Script complet :
1. Tracking multi-séquences pour plusieurs valeurs de DIST_MAX
2. Évaluation CLEAR-MOT (MOTA, MOTP, sensibilité) pour la première séquence
3. Tracé des courbes des métriques en fonction de DIST_MAX
Compatible pandas ≥ 2.0, Matplotlib ≥ 3.8
"""

import os
import math
import numpy as np
import pandas as pd
import cv2
import natsort
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os, cv2, natsort, numpy as np, pandas as pd
from tifffile import imread
from tqdm import tqdm
from lyso_tracker.lyso import Lyso
from lyso_tracker.descriptor import detect_particles_adapt
# ───────────────────────────────────────────────────────────────
# Réglages généraux
# ───────────────────────────────────────────────────────────────
FRAMES_ROOT      = "data_artificial/data_2_par20/data_frame_2"
AVI_OUT_ROOT     = "tracks_avi_multi"
CSV_OUT_ROOT     = "tracks_csv_multi"
FPS              = 5
PARTICLE_RAD     = 5      # pour detect_particles
BLOCK_SIZE       = 5
C_PARAM          = -5     # pour detect_particles
MAX_GAP          = 100    # gap-filling
DISTANCES_TO_TEST = [10,20, 30, 50, 100]

# Évaluation
GT_CSV            = "data_artificial/data_2_par20/data_csv_2/trajs_0_0.csv"
DIST_THRESHOLD    = 8     # seuil bipartite
FRAME_START_AT0   = True
OUT_PLOT_DIR      = "assoc_plots_multi"
os.makedirs(AVI_OUT_ROOT, exist_ok=True)
os.makedirs(CSV_OUT_ROOT, exist_ok=True)
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────
#  Fonctions utilitaires
# ───────────────────────────────────────────────────────────────
def normalize(path_img: str) -> np.ndarray | None:
    """Lit l’image en niveaux de gris et la normalise 0-255 (uint8)."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Impossible de lire {path_img}")
        return None
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def load_frames_and_detections(frames_dir: str,
                               radius_px: int,
                               block_size: int,
                               C: int) -> tuple[list, list]:
    """Retourne (detections, frames) pour un dossier *_frames."""
    img_names = natsort.natsorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".tiff", ".tif", ".png", ".jpeg", ".jpg"))
    ])
    frames, detections = [], []
    for name in img_names:
        frame = normalize(os.path.join(frames_dir, name))
        if frame is None:
            continue
        frames.append(frame)
        detections.append(
            detect_particles_adapt(frame,
                                   radius_px=radius_px,
                                   block_size=block_size,
                                   C=C)
        )
    return detections, frames


def track_one_sequence(frames_dir: str,
                       dist_max: int,
                       max_gap: int,
                       fps: int,
                       radius_px: int,
                       block_size: int,
                       C: int,
                       avi_out_root: str,
                       csv_out_root: str):
    """
    Tracking sur un dossier *_frames, sauve AVI et CSV.
    """
    stem = os.path.basename(frames_dir).replace("_frames", "")
    avi_out = os.path.join(avi_out_root, f"{stem}_d{dist_max}_tracked.avi")
    csv_out = os.path.join(csv_out_root, f"{stem}_d{dist_max}_tracked.csv")

    detections, frames = load_frames_and_detections(
        frames_dir, radius_px, block_size, C)
    if not frames:
        print(f"[SKIP] Aucun frame valide dans {frames_dir}")
        return

    tracker = Lyso()
    tracker.track(detections, dist_max, max_gap)
    tracker.draw_tracks(frames, detections, masks=None,
                        output_path=avi_out, fps=fps)
    tracker.save_csv(csv_out)
    print(f"{stem} (dist_max={dist_max}) → {avi_out}, {csv_out}")


def run_tracking(frames_root: str,
                 video_root: str,
                 csv_root: str,
                 distances: list[int],
                 fps: int = 5,
                 radius_px: int = 5,
                 block_size: int = 5,
                 C: int = -5,
                 max_gap: int = 100):
    """
    Exécute le tracking sur toutes les séquences pour chaque DIST_MAX.
    """
    # Prépare les dossiers
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)

    all_frame_dirs = [os.path.join(frames_root, d)
                      for d in os.listdir(frames_root)
                      if d.endswith("_frames")
                      and os.path.isdir(os.path.join(frames_root, d))]
    all_frame_dirs = natsort.natsorted(all_frame_dirs)
    print(f"{len(all_frame_dirs)} séquences trouvées dans '{frames_root}'")

    for dist in distances:
        print(f"\n-- Tracking avec DIST_MAX = {dist} --")
        avi_out_d = os.path.join(video_root, f"d{dist}")
        csv_out_d = os.path.join(csv_root, f"d{dist}")
        os.makedirs(avi_out_d, exist_ok=True)
        os.makedirs(csv_out_d, exist_ok=True)

        for seq_dir in tqdm(all_frame_dirs, desc=f"DIST_MAX={dist}"):
            track_one_sequence(
                frames_dir=seq_dir,
                dist_max=dist,
                max_gap=max_gap,
                fps=fps,
                radius_px=radius_px,
                block_size=block_size,
                C=C,
                avi_out_root=avi_out_d,
                csv_out_root=csv_out_d
            )
    print("\nTracking terminé pour toutes les séquences.")

# ───────────────────────────────────────────────────────────────
#  CLEAR-MOT Évaluation pour la première séquence
# ───────────────────────────────────────────────────────────────
def load_tracks_csv(path, id_col='id', frame_col='frame', x_col='x', y_col='y', frame_start_zero=True):
    df = pd.read_csv(path)
    if not frame_start_zero:
        df[frame_col] -= df[frame_col].min()
    tracks = []
    for _, g in df.groupby(id_col):
        g = g.sort_values(frame_col)
        f = g[frame_col].to_numpy(int)
        xy = g[[x_col, y_col]].to_numpy(float)
        arr = np.full((f.max() + 1, 2), np.nan)
        arr[f] = xy
        tracks.append(arr)
    return tracks


def extract_frame_and_pos(track_arr):
    frames = np.where(~np.isnan(track_arr[:, 0]))[0]
    return frames, track_arr[frames]


def evaluate_first_sequence(frames_root, csv_root, gt_csv, distances, dist_threshold, frame_start_at0):
    # GT tracks
    gt_tracks = load_tracks_csv(gt_csv, frame_start_zero=frame_start_at0)
    total_gt = sum(len(np.where(~np.isnan(tr[:,0]))[0]) for tr in gt_tracks)

    # Détermine le stem de la première séquence
    dirs = [d for d in os.listdir(frames_root) if d.endswith("_frames")]
    stem = natsort.natsorted(dirs)[0].replace("_frames", "")

    metrics = []
    for dist in distances:
        pred_path = os.path.join(csv_root, f"d{dist}", f"{stem}_d{dist}_tracked.csv")
        pr_tracks = load_tracks_csv(pred_path, frame_start_zero=frame_start_at0)

        fn = fp = id_sw = 0
        total_dist = pairs = 0
        hist = {}
        last_seen = {}

        # boucle temporelle
        T_max = max((np.where(~np.isnan(t[:,0]))[0].max() for t in gt_tracks + pr_tracks), default=0)
        for t in range(T_max + 1):
            gt_ids = [i for i, tr in enumerate(gt_tracks) if t < len(tr) and not np.isnan(tr[t,0])]
            pr_ids = [j for j, tr in enumerate(pr_tracks) if t < len(tr) and not np.isnan(tr[t,0])]
            gt_pos = np.array([gt_tracks[i][t] for i in gt_ids]) if gt_ids else np.empty((0,2))
            pr_pos = np.array([pr_tracks[j][t] for j in pr_ids]) if pr_ids else np.empty((0,2))

            # assignment
            matches = []
            if gt_ids and pr_ids:
                D = np.linalg.norm(gt_pos[:,None] - pr_pos[None,:,:], axis=2)
                r, c = linear_sum_assignment(D)
                for rr, cc in zip(r, c):
                    if D[rr, cc] < dist_threshold:
                        matches.append((gt_ids[rr], pr_ids[cc], D[rr, cc]))

            matched_gt = {m[0] for m in matches}
            matched_pr = {m[1] for m in matches}
            fn += len(gt_ids) - len(matched_gt)
            fp += len(pr_ids) - len(matched_pr)

            for gid, pid, dval in matches:
                total_dist += dval
                pairs += 1
                if (gid in hist and hist[gid] != pid and last_seen.get(gid) == t-1):
                    id_sw += 1
                hist[gid] = pid
                last_seen[gid] = t

        motp = total_dist / pairs if pairs else float('nan')
        mota = 100 * (1 - (fn + fp + id_sw) / max(total_gt,1))
        sens = 1 - fn / total_gt
        metrics.append({'dist_max': dist, 'MOTA(%)': mota, 'MOTP(px)': motp, 'sensibilité': sens})

    df = pd.DataFrame(metrics)
    # courbes
    fig, ax = plt.subplots()
    ax.plot(df['dist_max'], df['MOTA(%)'], marker='o')
    ax.set_xlabel('DIST_MAX (px)')
    ax.set_ylabel('MOTA (%)')
    ax.set_title('MOTA vs DIST_MAX')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOT_DIR, 'MOTA_vs_dist.png'), dpi=200)

    fig, ax = plt.subplots()
    ax.plot(df['dist_max'], df['MOTP(px)'], marker='o')
    ax.set_xlabel('DIST_MAX (px)')
    ax.set_ylabel('MOTP (px)')
    ax.set_title('MOTP vs DIST_MAX')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOT_DIR, 'MOTP_vs_dist.png'), dpi=200)

    fig, ax = plt.subplots()
    ax.plot(df['dist_max'], df['sensibilité'], marker='o')
    ax.set_xlabel('DIST_MAX (px)')
    ax.set_ylabel('Sensibilité')
    ax.set_title('Sensibilité vs DIST_MAX')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOT_DIR, 'sensibilité_vs_dist.png'), dpi=200)

    # Sauve résumé CSV
    summary_csv = os.path.join(OUT_PLOT_DIR, 'summary_metrics.csv')
    df.to_csv(summary_csv, index=False)
    print(f"Évaluation terminée. Fichiers dans {OUT_PLOT_DIR}")

# ───────────────────────────────────────────────────────────────
#  Point d'entrée
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Tracking
    run_tracking(
        FRAMES_ROOT,
        AVI_OUT_ROOT,
        CSV_OUT_ROOT,
        distances=DISTANCES_TO_TEST,
        fps=FPS,
        radius_px=PARTICLE_RAD,
        block_size=BLOCK_SIZE,
        C=C_PARAM,
        max_gap=MAX_GAP
    )

    # 2. Évaluation de la première séquence + tracés
    evaluate_first_sequence(
        FRAMES_ROOT,
        CSV_OUT_ROOT,
        GT_CSV,
        DISTANCES_TO_TEST,
        DIST_THRESHOLD,
        FRAME_START_AT0
    )
