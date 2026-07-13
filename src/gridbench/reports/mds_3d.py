import gridvis.display as di
from gridbench import store

import inspect
import pickle
import networkx as nx
import numpy as np
import argparse
from sklearn.manifold import MDS

import sys
import os
from contextlib import redirect_stdout
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import DistanceMetric

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


def load_data_from_file(filename):
    try:
        infile = open(filename, 'rb')
        data_dict = pickle.load(infile)
        print(os.path.basename(filename), "loaded", flush=True)
        return data_dict
    except FileNotFoundError:
        print("File {} not found, exiting.".format(filename), flush=True)
        sys.exit(1)
    except Exception as inst:
        print("exception: ", type(inst))
        print("args:", inst.args)
        print(inst, ", exiting.")
        sys.exit(1)


def calc_average_distance_matrix(matrix):
    return (matrix + matrix.T) / 2


def get_matrix_embedding(matrix, components=2):
    params = inspect.signature(MDS).parameters
    model_kwargs = {
        "n_components": components,
        "n_init": 4,
        "random_state": 1,
    }
    if "metric_mds" in params:
        model_kwargs["metric_mds"] = True
        model_kwargs["metric"] = "precomputed"
        if "init" in params:
            model_kwargs["init"] = "random"
    else:
        model_kwargs["metric"] = True
        model_kwargs["dissimilarity"] = "precomputed"
        if "normalized_stress" in params:
            model_kwargs["normalized_stress"] = "auto"
    model = MDS(**model_kwargs)
    return model.fit_transform(matrix)


def main():
    path = str(store.data_root())
    def _find_pickle(prefix):
        for root, _dirs, files in os.walk(path):
            pickles = sorted(
                file for file in files if file.endswith(".pickle") and file.startswith(prefix)
            )
            if pickles:
                return os.path.relpath(root, path), pickles[0]
        return None, None

    data_dir, filename = _find_pickle("collated-")
    if filename is None:
        data_dir, filename = _find_pickle("data-")

    if filename is None or data_dir is None:
        raise RuntimeError(f"No suitable pickle files found beneath {path}")

    data_dir_path = os.path.join(path, data_dir)
    data = load_data_from_file(os.path.join(path, data_dir, filename))

    env = data.get('env') or data['state_dist'].env
    shape = data.get('shape', env.shape)
    frees = data.get('frees')
    if frees is None:
        frees = data.get('free')
        if frees is not None and frees.ndim == 1:
            raise ValueError(
                f"Selected file '{filename}' contains a 1D free-energy vector; "
                "please select a collated dataset."
            )
    if frees is None:
        raise ValueError(f"Free energy data not found in '{filename}'")

    distance_matrix = calc_average_distance_matrix(frees)

    manhattan = data.get('manhattan', getattr(env, 'manhattan', True))
    if manhattan:
        metric = 'cityblock'
    else:
        metric = 'chebyshev'
    norm = DistanceMetric.get_metric(metric)
    pairwise_distance = norm.pairwise(distance_matrix)
    distance = distance_matrix


    coords = get_matrix_embedding(distance, components=3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colour_indices = di.select_state_colours(env.shape)
    node_colours = [di.GRAPH_CMAP(colour_indices[idx])[0:3] for idx in range(env.nS)]

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color=node_colours,
        s=880,
        edgecolors='gray',
        alpha=0.8,
        depthshade=True,
    )

    edges = set()
    for state in range(env.nS):
        for action in env.D[state]:
            for _, successor, _, _ in env.D[state][action]:
                if successor == state:
                    continue
                edge = tuple(sorted((state, successor)))
                edges.add(edge)

    for i, j in edges:
        xs = [coords[i, 0], coords[j, 0]]
        ys = [coords[i, 1], coords[j, 1]]
        zs = [coords[i, 2], coords[j, 2]]
        ax.plot(xs, ys, zs, color='gray', alpha=0.4, linewidth=1.0)

    for idx, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(idx), fontsize=12, ha='center', va='center', color='black', weight='bold')

    ax.set_axis_off()
    output_stem = os.path.splitext(os.path.basename(filename))[0]
    png_path = os.path.join(path, data_dir, f"{output_stem}.png")
    html_path = os.path.join(path, data_dir, f"{output_stem}.html")
    plt.tight_layout(pad=0.1)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.05)

    if go is not None:
        edge_x, edge_y, edge_z = [], [], []
        for i, j in edges:
            edge_x.extend([coords[i, 0], coords[j, 0], None])
            edge_y.extend([coords[i, 1], coords[j, 1], None])
            edge_z.extend([coords[i, 2], coords[j, 2], None])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none',
        )

        node_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=18,
                color=[mpl.colors.to_hex(col) for col in node_colours],
                line=dict(color='gray', width=1.5),
                opacity=0.95,
                symbol='circle',
            ),
            text=[str(idx) for idx in range(env.nS)],
            textposition='top center',
            textfont=dict(color='black', size=14),
            hoverinfo='text',
        )

        fig_plotly = go.Figure(data=[edge_trace, node_trace])
        fig_plotly.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        fig_plotly.write_html(html_path, include_plotlyjs='cdn')
    else:
        print("Plotly not installed; skipping interactive export.")

    plt.show(block=True)
    print('finished')


if __name__ == "__main__":
    main()
