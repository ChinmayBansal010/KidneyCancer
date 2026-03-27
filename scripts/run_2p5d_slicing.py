from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.slicing_2p5d.build_slices import build_all

if __name__ == "__main__":
    build_all()
