# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
# Ignacio Vizzo, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from tqdm import tqdm

from map_closures.datasets import eval_dataloaders, sequence_dataloaders
from map_closures.pybind import gt_closures_pybind


def name_callback(value: str):
    if not value:
        return value
    dl = eval_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_available_dl_help = eval_dataloaders()

docstring = f"""
Generate Ground Truth Loop Closure Indices\n
\b
Examples:\n
# Use a specific dataloader with groundtruth poses: apollo, kitti, mulran, ncd, helipr
$ gt_closure_pipeline mulran <path-to-mulran-sequence-root>
"""


@app.command(help=docstring)
def gt_closure_pipeline(
    dataloader: str = typer.Argument(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=eval_dataloaders,
        callback=name_callback,
        help="Use a specific dataloader from those supported by MapClosures",
    ),
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    # Aditional Options ---------------------------------------------------------------------------
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    max_range: Optional[float] = typer.Option(
        100.0,
        "--max_range",
        "-r",
        show_default=True,
        help="[Optional] Maximum range of sensor / Maximum distance between closures",
        rich_help_panel="Additional Options",
    ),
    overlap_threshold: Optional[float] = typer.Option(
        0.5,
        "--overlap",
        "-o",
        show_default=True,
        help="[Optional] Overlap Threshold between scans at closures",
        rich_help_panel="Additional Options",
    ),
    voxel_size: Optional[float] = typer.Option(
        0.5,
        "--voxel_size",
        "-v",
        show_default=True,
        help="[Optional] Voxel size for downsamlping scans",
        rich_help_panel="Additional Options",
    ),
):
    if dataloader in sequence_dataloaders() and sequence is None:
        print('[ERROR] You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    from map_closures.datasets import dataset_factory

    dataset = dataset_factory(
        dataloader=dataloader,
        data_dir=data,
        # Additional options
        sequence=sequence,
    )
    if hasattr(dataset, "gt_poses"):
        generate_gt_closures(dataset, max_range, overlap_threshold, voxel_size)
    else:
        print("[ERROR] Groundtruth poses not found")
        raise typer.Exit(code=1)


def run():
    app()


def generate_gt_closures(
    dataset, max_range: float, overlap_threshold: float = 0.5, voxel_size: float = 0.5
):
    base_dir = dataset.sequence_dir if hasattr(dataset, "sequence_dir") else ""
    os.makedirs(os.path.join(base_dir, "loop_closure"), exist_ok=True)
    file_path_closures = os.path.join(base_dir, "loop_closure", "gt_closures.txt")
    if os.path.exists(file_path_closures):
        closures = np.loadtxt(file_path_closures, dtype=int)
        print(f"[INFO] Found closure ground truth at {file_path_closures}")

    else:
        print("[INFO] Computing Ground Truth Closures!")
        sampling_distance = 2.0
        gt_closures_pipeline = gt_closures_pybind._GTClosures(
            len(dataset), sampling_distance, overlap_threshold, voxel_size, max_range
        )
        for idx in tqdm(range(len(dataset)), "Loading Data ..."):
            try:
                points, _ = dataset[idx]
            except ValueError:
                points = dataset[idx]
            points = points[np.where(np.linalg.norm(points, axis=1) < max_range)[0]]
            gt_closures_pipeline._AddPointCloud(
                idx, gt_closures_pybind._Vector3dVector(points), dataset.gt_poses[idx]
            )
        num_query_segments = gt_closures_pipeline._GetSegments()
        closures = []
        for query_segment_idx in tqdm(
            range(num_query_segments), "Computing Groundtruth Closures ..."
        ):
            closures.append(
                np.asarray(
                    gt_closures_pipeline._ComputeClosuresForQuerySegment(query_segment_idx), int
                )
            )
        closures = np.vstack(closures)
        np.savetxt(file_path_closures, closures)

    return closures
