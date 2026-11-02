
import datetime
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


def yaml_to_argv(config: str | Path) -> list[str]:
    argv: list[str] = []

    for key, value in config.items():
        argument = f"--{key}"

        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                argv.append(argument)
            continue

        if isinstance(value, list):
            argv.append(argument)
            argv.extend(map(str, value))
            continue

        argv.extend([argument, str(value)])

    return argv


def constrained_km(data, n_clusters=10):
    # Convert torch tensor to numpy
    x = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    
    # 1. Standard KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=10, n_init=10, random_state=42)
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_

    # 2. Greedy constraint enforcement (Caps max size to protect limits)
    size_max = n_clusters * 6
    counts = np.bincount(labels, minlength=n_clusters)
    
    if np.any(counts > size_max):
        distances = np.linalg.norm(x[:, np.newaxis] - centers, axis=2)
        sorted_points = np.argsort(np.min(distances, axis=1))  # Process closest points first
        
        counts = np.zeros(n_clusters, dtype=int)
        for idx in sorted_points:
            # Assign to closest cluster that isn't full
            allowed_clusters = np.where(counts < size_max)[0]
            best_cluster = allowed_clusters[np.argmin(distances[idx, allowed_clusters])]
            labels[idx] = best_cluster
            counts[best_cluster] += 1

    return torch.from_numpy(centers), labels.tolist()


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur



