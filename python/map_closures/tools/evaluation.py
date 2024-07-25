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
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.linalg import inv, norm
from rich import box
from rich.console import Console
from rich.table import Table


@dataclass
class LocalMap:
    pointcloud: np.ndarray
    density_map: np.ndarray
    scan_indices: np.ndarray
    scan_poses: np.ndarray


def compute_closure_indices(
    ref_indices: np.ndarray,
    query_indices: np.ndarray,
    ref_scan_poses: np.ndarray,
    query_scan_poses: np.ndarray,
    relative_tf: np.ndarray,
    distance_threshold: float,
):
    T_query_world = inv(query_scan_poses[0])
    T_ref_world = inv(ref_scan_poses[0])

    # bring all poses to a common frame at the query map
    query_locs = (T_query_world @ query_scan_poses)[:, :3, -1].squeeze()
    ref_locs = (relative_tf @ T_ref_world @ ref_scan_poses)[:, :3, -1].squeeze()

    closure_indices = []
    query_id_start = query_indices[0]
    ref_id_start = ref_indices[0]
    qq, rr = np.meshgrid(query_indices, ref_indices)
    distances = norm(query_locs[qq - query_id_start] - ref_locs[rr - ref_id_start], axis=2)
    ids = np.where(distances < distance_threshold)
    for r_id, q_id in zip(ids[0] + ref_id_start, ids[1] + query_id_start):
        closure_indices.append((r_id, q_id))
    closure_indices = set(map(lambda x: tuple(sorted(x)), closure_indices))
    return closure_indices


class EvaluationMetrics:
    def __init__(self, true_positives, false_positives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = np.nan

        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = np.nan

        try:
            self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            self.F1 = np.nan


class StubEvaluation(ABC):
    def __init__(self):
        pass

    def print(self):
        pass

    def append(self, *kwargs):
        pass

    def compute_closures_and_metrics(self):
        pass

    def log_to_file(self, *kwargs):
        pass


class EvaluationPipeline(StubEvaluation):
    def __init__(
        self,
        gt_closures: np.ndarray,
        dataset_name: str,
        closure_distance_threshold: float,
    ):
        self._dataset_name = dataset_name
        self._closure_distance_threshold = closure_distance_threshold

        self.closure_indices_list: List[Set[Tuple]] = []
        self.inliers_count_list: List = []

        self.predicted_closures: Dict[int, Set[Tuple[int]]] = {}
        self.metrics: Dict[int, EvaluationMetrics] = {}

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self):
        self._log_to_console()

    def append(
        self,
        ref_local_map: LocalMap,
        query_local_map: LocalMap,
        relative_pose: np.ndarray,
        distance_threshold: float,
        inliers_count: int,
    ):
        closure_indices = compute_closure_indices(
            ref_local_map.scan_indices,
            query_local_map.scan_indices,
            ref_local_map.scan_poses,
            query_local_map.scan_poses,
            relative_pose,
            distance_threshold,
        )
        self.closure_indices_list.append(closure_indices)
        self.inliers_count_list.append(inliers_count)

    def compute_closures_and_metrics(
        self,
    ):
        print("[INFO] Computing Loop Closure Evaluation Metrics")
        for inliers_threshold in range(5, 15):
            closures = set()
            for closure_indices, inliers_count in zip(
                self.closure_indices_list, self.inliers_count_list
            ):
                if inliers_count >= inliers_threshold:
                    closures = closures.union(closure_indices)

            tp = len(self.gt_closures.intersection(closures))
            fp = len(closures) - tp
            fn = len(self.gt_closures) - tp
            self.metrics[inliers_threshold] = EvaluationMetrics(tp, fp, fn)
            self.predicted_closures[inliers_threshold] = closures

    def _rich_table_pr(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        table = Table(box=table_format)
        table.caption = f"Loop Closure Evaluation Metrics\n"
        table.add_column("RANSAC #Inliers", justify="center", style="cyan")
        table.add_column("True Positives", justify="center", style="magenta")
        table.add_column("False Positives", justify="center", style="magenta")
        table.add_column("False Negatives", justify="center", style="magenta")
        table.add_column("Precision", justify="left", style="green")
        table.add_column("Recall", justify="left", style="green")
        table.add_column("F1 score", justify="left", style="green")
        for [inliers, metric] in self.metrics.items():
            table.add_row(
                f"{inliers}",
                f"{metric.tp}",
                f"{metric.fp}",
                f"{metric.fn}",
                f"{metric.precision:.4f}",
                f"{metric.recall:.4f}",
                f"{metric.F1:.4f}",
            )
        return table

    def _log_to_console(self):
        console = Console()
        console.print(self._rich_table_pr())

    def log_to_file(self, results_dir):
        with open(os.path.join(results_dir, "evaluation_metrics.txt"), "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            table = self._rich_table_pr(table_format=box.ASCII_DOUBLE_HEAD)
            console.print(table)
        np.save(os.path.join(results_dir, "scan_level_closures.npy"), self.predicted_closures)
