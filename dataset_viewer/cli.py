import argparse
from pathlib import Path

import fiftyone as fo


class DataViewerNamespace(argparse.Namespace):
    dataset_root: Path
    dataset_name: str
    port: int


def parse_args() -> DataViewerNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset_root', type=Path, required=True, help='Path to dataset root',
    )
    parser.add_argument(
        '-n', '--dataset_name', default='Unknown dataset name', help='Dataset name',
    )
    parser.add_argument(
        '-p', '--port', type=int, default=5000, help='Port for running web app',
    )

    return parser.parse_args(namespace=DataViewerNamespace())


def main() -> int:
    args = parse_args()

    dataset = fo.Dataset.from_dir(
        dataset_dir=str(args.dataset_root),
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        name=args.dataset_name,
    )

    session = fo.launch_app(dataset, port=args.port)
    session.wait()

    return 0
