"""
poetry run python dsu/train/main.py --cfg-file dsu/train/config/0.json
"""

import click
import wandb

from dsu.train.callbacks import get_callbacks
from dsu.train.config import Config
from dsu.train.data import get_data
from dsu.train.model import get_model


@click.command()
@click.option("--cfg-file", type=click.STRING)
def main(cfg_file: str):
    cfg = Config.from_file(cfg_file)
    for i in range(cfg.num_runs):
        if cfg.use_wandb:
            wandb.init(project=cfg.wandb_project_name)
        ds, step_cfg = get_data(cfg)
        model = get_model(cfg)
        model.fit(
            ds.train,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=get_callbacks(wandb.run.name),
            steps_per_epoch=step_cfg.steps_per_epoch,
            validation_data=ds.val,
            validation_steps=step_cfg.validation_steps,
        )
        score = model.evaluate(ds.test)
        print(f"Test loss: {score[0]:.4f} / Test accuracy: {score[1]:.4%}")
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
