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
from typing import List, Set, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table


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

    def compute_metrics(self):
        pass

    def log_to_file(self, *kwargs):
        pass


class EvaluationPipeline(StubEvaluation):
    def __init__(
        self,
        gt_closures: np.ndarray,
        dataset_name: str,
    ):
        self._dataset_name = dataset_name

        self.closure_list: List[Tuple[int]] = []
        self.inliers_count_list: List[int] = []

        self.metrics = []

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self):
        self._log_to_console()

    def append(self, source_id: int, target_id: int, inliers_count: int):
        self.closure_list.append((source_id, target_id))
        self.inliers_count_list.append(inliers_count)

    def compute_metrics(self):
        print("[INFO] Computing Loop Closure Evaluation Metrics")
        for inliers_threshold in range(3, np.max(self.inliers_count_list)):
            closures = set()
            for closure_indices, inliers_count in zip(self.closure_list, self.inliers_count_list):
                if inliers_count >= inliers_threshold:
                    closures.add(closure_indices)

            tp = len(self.gt_closures.intersection(closures))
            fp = len(closures) - tp
            fn = len(self.gt_closures) - tp
            self.metrics.append(EvaluationMetrics(tp, fp, fn))

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
        for i, metric in enumerate(self.metrics):
            table.add_row(
                f"{i + 3} {metric.tp} {metric.fp} {metric.fn} {metric.precision:.4f} {metric.recall:.4f} {metric.F1:.4f}"
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
        np.savetxt(
            os.path.join(results_dir, "closure_indices.txt"), np.concatenate(self.closure_list)
        )
