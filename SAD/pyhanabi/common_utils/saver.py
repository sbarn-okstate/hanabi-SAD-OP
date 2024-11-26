import os
import tensorflow as tf
import numpy as np


class TopkSaver:
    def __init__(self, save_dir, topk):
        self.save_dir = save_dir
        self.topk = topk
        self.worst_perf = -float("inf")
        self.worst_perf_idx = 0
        self.perfs = [self.worst_perf]

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model, perf):
        """
        Save the model if its performance is within the top-k.

        Args:
            model (tf.keras.Model): The model to be saved.
            perf (float): The performance metric of the model.

        Returns:
            bool: True if the model was saved, False otherwise.
        """
        if perf <= self.worst_perf:
            # The performance is not within the top-k; skip saving.
            return False

        # Save the model in the designated save slot
        model_name = f"model{self.worst_perf_idx}.h5"
        model_path = os.path.join(self.save_dir, model_name)
        model.save(model_path)

        if len(self.perfs) < self.topk:
            # If there is still room in the top-k list, add the new performance.
            self.perfs.append(perf)
        else:
            # Replace the worst performing model in the top-k list.
            self.perfs[self.worst_perf_idx] = perf

        # Update the worst performance in the top-k list
        self.worst_perf = self.perfs[0]
        self.worst_perf_idx = 0
        for i, performance in enumerate(self.perfs):
            if performance < self.worst_perf:
                self.worst_perf = performance
                self.worst_perf_idx = i

        return True
