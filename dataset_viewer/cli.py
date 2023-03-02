import argparse
import csv
from pathlib import Path

import fiftyone as fo
from tqdm import tqdm


class DataViewerNamespace(argparse.Namespace):
    dataset_file: Path
    dataset_name: str
    port: int
    threshold: int
    undetected: bool


def parse_args() -> DataViewerNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--dataset_file', type=Path, required=True, help='Path to dataset .csv file',
    )
    parser.add_argument(
        '-n', '--dataset_name', default='Unknown dataset name', help='Dataset name',
    )
    parser.add_argument(
        '-p', '--port', type=int, default=5000, help='Port for running web app',
    )
    parser.add_argument(
        '-t', '--threshold', type=int, default=3, help='Minimal number of images for person',
    )
    parser.add_argument(
        '-u', '--undetected',
        action='store_true',
        help='Flag to view undetected .csv file',
        default=False,
    )

    return parser.parse_args(namespace=DataViewerNamespace())


def main() -> int:
    args = parse_args()
    samples = []

    if args.undetected:
        with open(args.dataset_file, 'r', newline='', encoding='utf8') as input_:
            csv_reader = csv.reader(input_, delimiter=',')

            for row in csv_reader:
                image_path, *__ = row
                path = args.dataset_file.parent / image_path
                sample = fo.Sample(filepath=path.as_posix())
                samples.append(sample)
    else:
        with open(args.dataset_file, 'r', newline='', encoding='utf8') as input_:
            csv_dictreader = csv.DictReader(input_, delimiter=',')

            for dict_row in tqdm(csv_dictreader, desc='Iterating through csv file'):
                if int(dict_row.get('number_of_images_per_person_id', 0)) < args.threshold:
                    continue

                sex = None
                sex_str = dict_row.get('sex', None)
                if sex_str is not None:
                    sex = 'male' if sex_str == '2' else 'female'

                path = args.dataset_file.parent / dict_row.get('image_path', 'nan')
                sample = fo.Sample(path.as_posix())
                sample['ground_truth'] = fo.Classification(label=dict_row.get('person_id', 'nan'))
                sample['city'] = dict_row.get('city', 'nan')
                sample['age'] = dict_row.get('age', 'nan')
                sample['sex'] = sex or 'nan'
                sample['vk_id'] = dict_row.get('vk_id', 'nan')
                sample['trashness'] = float(dict_row.get('trashness', 'nan'))

                samples.append(sample)

    dataset = fo.Dataset(args.dataset_name)
    dataset.add_samples(samples)

    session = fo.launch_app(dataset, port=args.port)
    session.wait()

    return 0
