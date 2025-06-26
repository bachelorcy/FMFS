import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from metrics import Metric

class Evaluator:
    def __init__(self, pred_dir, gt_dir):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.metric = Metric()

    def evaluate(self):
        image_names = os.listdir(self.pred_dir)
        for name in tqdm(image_names, desc="Evaluating", unit="img"):
            pred_path = os.path.join(self.pred_dir, name)
            gt_name = name
            gt_path = os.path.join(self.gt_dir, gt_name)

            pred_img = Image.open(pred_path).convert("L")
            gt_img = Image.open(gt_path).convert("L")

            pred_np = np.array(pred_img)
            gt_np = np.array(gt_img)


            pred_bin = (pred_np > 127).astype(np.uint8)
            gt_bin = (gt_np > 127).astype(np.uint8)

            self.metric.step(pred=pred_bin, gt=gt_bin)

    def get_results(self):
        return self.metric.get_results(bit_width=4)


if __name__ == "__main__":
    config = {
        'datasets': ['CAMO', 'COD10K', 'NC4K'],
        'base_dir': '../FMFS-Net/data/Test/',
        'save_dir': '../FMFS-Net/Results',
    }

    total_results = {}
    for dataset in config['datasets']:
        print(f"Processing dataset: {dataset}")
        pred_dir = os.path.join(config['save_dir'], dataset)
        gt_dir = os.path.join(config['base_dir'], dataset, 'GT')

        evaluator = Evaluator(pred_dir=pred_dir, gt_dir=gt_dir)
        evaluator.evaluate()
        results = evaluator.get_results()
        total_results[dataset] = results

        log_str = f"[{dataset}] Evaluation Results: | "
        log_str += " | ".join([f"{k}: {v}" for k, v in results.items()])
        print(log_str)

    print("Evaluation completed for all datasets.")