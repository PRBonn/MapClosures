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
from pathlib import Path
from typing import Optional

import typer

from map_closures.datasets import available_dataloaders


def version_callback(value: bool):
    if value:
        import map_closures
        print(f"MapClosures Version: {map_closures.__version__}")
        raise typer.Exit(0)


def name_callback(value: str):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_available_dl_help = available_dataloaders()

docstring = f"""
Effectively Detecting Loop Closures using Point Cloud Density Maps\n
\b
Examples:\n
# Use a specific dataloader: apollo, kitti, mulran, ncd, helipr
$ map_closure_pipeline --dataloader mulran --config <path-to-config-file> <path-to-mulran-sequence-root> <path-to-store-results>
"""


@app.command(help=docstring)
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
        help="[Optional] Use a specific dataloader from those supported by KISS-ICP",
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
        print(f"[ERRROR] Please provide a dataloader")
        raise typer.Exit(1)

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
    ).run()


def run():
    app()
