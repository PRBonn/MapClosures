# MIT License
#
# Copyright (c) 2026 Saurabh Gupta
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
import glob
import os
from pathlib import Path
from typing import Optional

import typer

from map_closures.datasets import (
    available_dataloaders,
    eval_dataloaders,
    sequence_dataloaders,
)


def guess_dataloader(data: Path, default_dataloader: str):
    """Guess which dataloader to use in case the user didn't specify with --dataloader flag."""
    if data.is_file():
        if data.name == "metadata.yaml":
            return "rosbag", data.parent  # database is in directory, not in .yml
        if data.name.split(".")[-1] in "bag":
            return "rosbag", data
        if data.name.split(".")[-1] == "pcap":
            return "ouster", data
        if data.name.split(".")[-1] == "mcap":
            return "mcap", data
    elif data.is_dir():
        if (data / "metadata.yaml").exists():
            # a directory with a metadata.yaml must be a ROS2 bagfile
            return "rosbag", data
        if len(glob.glob(os.path.join(data, "*.bag"))) > 0:
            return "rosbag", data
    return default_dataloader, data


def version_callback(value: bool):
    if value:
        try:
            # Check that the python bindings are properly built and can be loaded at runtime
            from map_closures.pybind import map_closures_pybind
        except ImportError as e:
            print(f"[ERRROR] Python bindings not properly built!")
            print(f"[ERRROR] '{e}'")
            raise typer.Exit(1)

        import map_closures

        print(f"[INFO] MapClosures Version: {map_closures.__version__}")
        raise typer.Exit(0)


def name_callback(value: str):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app_intra_session = typer.Typer(add_completion=False, rich_markup_mode="rich")

_available_dl_help = available_dataloaders()
_available_dl_help.remove("generic")
_available_dl_help.remove("mcap")
_available_dl_help.remove("ouster")
_available_dl_help.remove("rosbag")

docstring = f"""
Effectively Detecting Loop Closures using Point Cloud Density Maps\n
\b
Examples:\n
# Use a specific dataloader: apollo, kitti, mulran, ncd, helipr
$ map_closure_pipeline --dataloader mulran --config <path-to-config-file> <path-to-mulran-sequence-root> <path-to-store-results>
"""


@app_intra_session.command(help=docstring)
def map_closure_pipeline(
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported by MapClosures",
    ),
    results_dir: Path = typer.Argument(
        ...,
        help="The path where results are to be stored",
        show_default=False,
        exists=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    eval: Optional[bool] = typer.Option(
        False,
        "--eval",
        "-e",
        rich_help_panel="Additional Options",
        help="[Optional] Run loop closure evaluation",
    ),
    vis: Optional[bool] = typer.Option(
        False,
        "--vis",
        "-v",
        rich_help_panel="Additional Options",
        help="[Optional] Visualization of closures",
    ),
    opt: Optional[bool] = typer.Option(
        False,
        "--opt",
        "-o",
        rich_help_panel="Additional Options",
        help="[Optional] Run pgo optimization with detected closures",
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
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        show_default=False,
        help="[Optional] Only valid when processing rosbag files",
        rich_help_panel="Additional Options",
    ),
    meta: Optional[Path] = typer.Option(
        None,
        "--meta",
        "-m",
        exists=True,
        show_default=False,
        help="[Optional] For Ouster pcap dataloader, specify metadata json file path explicitly",
        rich_help_panel="Additional Options",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show the current version of MapClosures",
        callback=version_callback,
        is_eager=True,
    ),
):
    if not dataloader:
        dataloader, data = guess_dataloader(data, default_dataloader="generic")

    if dataloader in sequence_dataloaders() and sequence is None:
        print('[ERROR] You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    if eval is True and dataloader not in eval_dataloaders():
        print(
            "[WARNING] Cannot run evaluation pipeline for this dataloader, no groundtruth poses available"
        )
        print("[WARNING] Turning off eval mode!!")
        eval = False

    from map_closures.datasets import dataset_factory
    from map_closures.pipeline import MapClosurePipeline

    MapClosurePipeline(
        dataset=dataset_factory(
            dataloader=dataloader,
            data_dir=data,
            # Additional options
            sequence=sequence,
            topic=topic,
            meta=meta,
        ),
        config_path=config,
        results_dir=results_dir,
        eval=eval,
        vis=vis,
        opt=opt,
    ).run().print()


app_inter_session = typer.Typer(add_completion=False, rich_markup_mode="rich")

docstring = f"""
Multi-Session Loop Closure Detection\n
\b
Examples:\n
# Detect closures between two sessions (query vs reference)
$ inter_session_map_closure_pipeline kitti <path-to-query-data> <path-to-ref-results-dir> <path-to-store-results>
"""


@app_inter_session.command(name="inter_session_map_closure_pipeline", help=docstring)
def inter_session_map_closure_pipeline(
    dataloader_query: str = typer.Argument(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="Dataloader for the query session",
    ),
    data_query: Path = typer.Argument(
        ...,
        help="Data directory for the query session",
        show_default=False,
    ),
    ref_results_dir: Path = typer.Argument(
        ...,
        help="Path to the reference session results directory (output of map_closure_pipeline with write_data_to_disk())",
        show_default=False,
    ),
    results_dir: Path = typer.Argument(
        ...,
        help="Path where inter-session results are to be stored",
        show_default=False,
        exists=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    eval: Optional[bool] = typer.Option(
        False,
        "--eval",
        "-e",
        rich_help_panel="Additional Options",
        help="[Optional] Run the Evaluation Pipeline",
    ),
    vis: Optional[bool] = typer.Option(
        False,
        "--vis",
        "-v",
        rich_help_panel="Additional Options",
        help="[Optional] Visualization of closures",
    ),
    opt: Optional[bool] = typer.Option(
        False,
        "--opt",
        "-o",
        rich_help_panel="Additional Options",
        help="[Optional] Run pgo optimization with detected closures",
    ),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        show_default=False,
        help="[Optional] Only valid when processing rosbag files",
        rich_help_panel="Additional Options",
    ),
    meta: Optional[Path] = typer.Option(
        None,
        "--meta",
        "-m",
        exists=True,
        show_default=False,
        help="[Optional] For Ouster pcap dataloader, specify metadata json file path explicitly",
        rich_help_panel="Additional Options",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show the current version of MapClosures",
        callback=version_callback,
        is_eager=True,
    ),
):
    if not dataloader_query:
        dataloader_query, data_query = guess_dataloader(data_query, default_dataloader="generic")

    if dataloader_query in sequence_dataloaders() and sequence is None:
        print('[ERROR] You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    from map_closures.datasets import dataset_factory
    from map_closures.inter_session_pipeline import InterSessionMapClosurePipeline

    InterSessionMapClosurePipeline(
        dataset_query=dataset_factory(
            dataloader=dataloader_query,
            data_dir=data_query,
            sequence=sequence,
            topic=topic,
            meta=meta,
        ),
        ref_data_path=ref_results_dir,
        config_path=config,
        results_dir=results_dir,
        eval=eval,
        vis=vis,
        opt=opt,
    ).run()


def run_intra_session():
    app_intra_session()


def run_inter_session():
    app_inter_session()
