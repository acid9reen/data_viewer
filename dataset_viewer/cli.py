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
                sample = fo.Sample(filepath=str(args.dataset_file.parent / image_path))
                samples.append(sample)
    else:
        with open(args.dataset_file, 'r', newline='', encoding='utf8') as input_:
            csv_reader = csv.reader(input_, delimiter=',')
            # Skip header
            next(csv_reader)

            for row in tqdm(csv_reader, desc='Iterating through csv file'):
                (
                    image_path,
                    person_id,
                    vk_id,
                    city,
                    age,
                    sex,
                    liveness_str,
                    number_of_images_per_person_id_str,
                    *__,
                ) = row

                if int(number_of_images_per_person_id_str) < args.threshold:
                    continue

                sample = fo.Sample(filepath=str(args.dataset_file.parent / image_path))
                sample['ground_truth'] = fo.Classification(label=person_id)
                sample['city'] = city
                sample['age'] = age
                sample['sex'] = 'male' if sex == '2' else 'female'
                sample['vk_id'] = vk_id
                sample['liveness'] = float(liveness_str)

                samples.append(sample)

    dataset = fo.Dataset(args.dataset_name)
    dataset.add_samples(samples)

    session = fo.launch_app(dataset, port=args.port)
    session.wait()

    return 0
