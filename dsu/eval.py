"""
poetry run python dsu/eval.py \
--model-path models/comfy-bird-187/176-0.9117 \
--ensemble-path models/comfy-bird-187/176-0.9117 \
--ensemble-path models/dainty-capybara-188/193-0.9081 \
--ensemble-path models/olive-night-189/147-0.9093 \
--ensemble-path models/sweet-valley-190/104-0.9097 \
--ensemble-path models/fearless-firebrand-186/101-0.9040 \
--output-path ./results \
--cifar-c-dir ~/Downloads/CIFAR-10-C
"""
from pathlib import Path
from typing import Tuple, Union, Dict, List, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import click
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from robustness_metrics.metrics.uncertainty import ExpectedCalibrationError

sns.set_style("whitegrid")


@click.command()
@click.option("--model-path", type=click.STRING)
@click.option("--ensemble-path", type=click.STRING, multiple=True)
@click.option("--cifar-c-dir", type=click.STRING)
@click.option("--output-path", type=click.STRING)
def main(model_path: str, ensemble_path: Tuple[str], cifar_c_dir: str, output_path: str):
    (train_images, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    print("Loading models..")
    models = get_models(ensemble_path, model_path)

    acc = tf.keras.metrics.Accuracy()
    ece = ExpectedCalibrationError(num_bins=10)
    results = []

    # get level 0, baseline results without corruptions
    for model_name, model_fn in models.items():
        preds = model_fn(preprocess_images(test_images))
        probs = tf.nn.softmax(preds, axis=1)
        acc.update_state(test_labels.flatten(), np.argmax(preds, axis=1))
        ece.add_batch(probs, label=test_labels)
        results.append(
            {
                "model_name": model_name,
                "corruption": "",
                "shift_intensity": 0,
                "accuracy": acc.result().numpy(),
                "ece": ece.result()["ece"],
            }
        )
        acc.reset_state()
        ece.reset_states()

    # iterate over corruptions
    corruptions = [cor_dir for cor_dir in Path(cifar_c_dir).iterdir() if cor_dir.stem != "labels"]
    for corruption in tqdm(corruptions, position=0):
        imgs = np.load(str(corruption))
        for severity in tqdm(range(0, 5), total=5, position=1):
            for model_name, model_fn in models.items():
                imgs_level = preprocess_images(imgs[severity * 10000: (severity + 1) * 10000])
                preds = model_fn(imgs_level)
                probs = tf.nn.softmax(preds, axis=1)
                acc.update_state(test_labels.flatten(), np.argmax(preds, axis=1))
                ece.add_batch(probs, label=test_labels)
                results.append(
                    {
                        "model_name": model_name,
                        "corruption": corruption.stem,
                        "shift_intensity": severity + 1,
                        "accuracy": acc.result().numpy(),
                        "ece": ece.result()["ece"],
                    }
                )
                acc.reset_state()
                ece.reset_states()

    save_results(output_path, results)


def get_models(ensemble_path, model_path) -> Dict[str, Callable]:
    model = tf.keras.models.load_model(model_path)
    ensemble = [tf.keras.models.load_model(model_path) for model_path in ensemble_path]
    ensemble_fn = lambda x: np.mean([model(x) for model in ensemble], axis=0)  # noqa
    models = {"vanilla": model, "ensemble": ensemble_fn}
    return models


def save_results(output_path, results: List[Dict[str, Union[int, float]]]):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path / "results.csv", index=False)
    for metric in ["ece", "accuracy"]:
        boxplot(df, metric).savefig(output_path / f"{metric}.png")


def boxplot(df: pd.DataFrame, metric: str) -> Figure:
    fig = plt.figure(dpi=100)
    sns.boxplot(data=df, x="shift_intensity", y=metric, hue="model_name")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig


def preprocess_images(imgs: np.ndarray) -> np.ndarray:
    imgs = tf.keras.applications.resnet.preprocess_input(imgs)
    imgs = imgs / 255.0
    return imgs


if __name__ == "__main__":
    main()
