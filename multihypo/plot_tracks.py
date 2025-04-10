import os
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_2d_tracks(input_csv, Flag_Save=False, Frames=None):
    """Plot tracks from a file in CSV format."""
    filename = os.path.basename(input_csv)
    plot_title = f"Tracks from {filename}"

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    lines = lines[1:]  # Skip header
    lines.sort(key=lambda x: int(x.split(',')[1]))  # Sort by track ID

    tracks = []
    track_ids = []
    previous_track_id = None
    for line in lines:
        current_track_id = int(line.split(',')[1])
        x, y = line.split(',')[2:]
        if current_track_id != previous_track_id:
            if previous_track_id is not None:
                tracks.append(track)
            track = []
            previous_track_id = current_track_id
            track_ids.append(previous_track_id)

        x, y = line.split(',')[2:]
        frame = line.split(',')[0]
        if 'None' in x:
            x = 'nan'
        if 'None' in y:
            y = 'nan'
        track.append((int(frame.strip()), float(x.strip()), float(y.strip())))
    tracks.append(track)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    colors = itertools.cycle(plt.cm.tab10.colors)

    for i, track in enumerate(tracks):
        color = next(colors)
        frame, x, y = zip(*track)

        if Frames is not None:
            j = np.where(np.all([np.array(frame) >= Frames[0], np.array(frame) <= Frames[1]], axis=0))[0]
            if np.size(j) >= 1:
                frame = np.array(frame)[j]
                x = np.array(x)[j]
                y = np.array(y)[j]
                axes[0].plot(x, y, color=color, label=f"Track {track_ids[i]}")
                axes[0].scatter(x, y, color=color)
                axes[1].plot(frame, x, '-o', color=color, label=f"x, Track {track_ids[i]}")
                axes[1].plot(frame, y, '--s', color=color, label=f"y, Track {track_ids[i]}")
        else:
            axes[0].plot(x, y, color=color, label=f"Track {track_ids[i]}")
            axes[0].scatter(x, y, color=color)
            axes[1].plot(frame, x, '-o', color=color, label=f"x, Track {track_ids[i]}")
            axes[1].plot(frame, y, '--s', color=color, label=f"y, Track {track_ids[i]}")

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(plot_title)
    axes[0].legend()

    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('X or Y')
    axes[1].legend()

    plt.tight_layout()

    if Flag_Save:
        if Frames is not None:
            fname_out = os.path.join(os.path.dirname(input_csv), f"{os.path.basename(input_csv).split('.')[0]}_Frames_{Frames[0]}_{Frames[1]}")
        else:
            fname_out = os.path.join(os.path.dirname(input_csv), os.path.basename(input_csv).split('.')[0])
        plt.savefig(f"{fname_out}.png")
    else:
        plt.show()

    plt.close()