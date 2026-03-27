from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.patch_extraction.extract_patches import extract_all

if __name__ == "__main__":
    extract_all()
