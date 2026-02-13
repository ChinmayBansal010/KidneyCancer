import yaml
import logging
import torch.multiprocessing as mp
from src.train.train_segmentation import run_training


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    cfg = yaml.safe_load(open("src/config/seg_config.yaml"))
    run_training(cfg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
