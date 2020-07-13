"""Function to train a model."""
import os
from time import time
from importlib.util import find_spec
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
from text_recognizer.datasets.dataset import Dataset
from text_recognizer.models.base import Model

if find_spec("text_recognizer") is None:
    import sys
    sys.path.append('.')

EARLY_STOPPING = True
MODEL_CHECKPOINT = True


class WandbImageLogger(Callback):
    """Custom callback for logging image predictions"""

    def __init__(self, model_wrapper: Model, dataset: Dataset, example_count: int = 4):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.val_images = dataset.x_test[:example_count]  # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        images = [
            wandb.Image(image, caption="{}: {}".format(*self.model_wrapper.predict_on_image(image)))
            for i, image in enumerate(self.val_images)
        ]
        wandb.log({"examples": images}, commit=False)


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, use_wandb: bool = False) -> Model:
    """Train model."""
    callbacks = []

    if MODEL_CHECKPOINT:
        model_checkpoint = ModelCheckpoint(filepath=os.path.join(wandb.run.dir, model.weights_filename_only), verbose=1)
        callbacks.append(model_checkpoint)

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
        callbacks.append(early_stopping)

    if use_wandb:
        image_callback = WandbImageLogger(model, dataset)
        wandb_callback = WandbCallback(save_model=True)
        callbacks.append(image_callback)
        callbacks.append(wandb_callback)

    model.network.summary()
    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, initial_epoch=wandb.run.step if wandb.run.resumed else 0, epochs=epochs, callbacks=callbacks)  # pylint: disable=line-too-long
    print("Training took {:2f}s".format(time() - t))

    return model
