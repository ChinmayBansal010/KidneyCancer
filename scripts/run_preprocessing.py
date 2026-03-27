from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.preprocessing.preprocess_dataset import preprocess_all

if __name__ == "__main__":
    preprocess_all()
