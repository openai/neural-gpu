"""Start and train the NeuralGPU.

See neuralgpu/trainer.py for flags and more information.
"""

import tensorflow as tf
from neuralgpu import trainer


def main(_):
  trainer.start_and_train()

if __name__ == "__main__":
  tf.app.run()
