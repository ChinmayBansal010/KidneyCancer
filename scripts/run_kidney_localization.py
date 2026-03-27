from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.kidney_localization.localize_dataset import localize_all

if __name__ == "__main__":
    localize_all()
