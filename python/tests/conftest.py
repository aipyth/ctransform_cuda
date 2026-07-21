import sys, glob, pathlib


def _find_build_dir():
    root = pathlib.Path(__file__).resolve().parents[2]      # repo root
    matches = glob.glob(str(root / "build" / "ctransform_cuda_py*.so"))
    if not matches:
        raise RuntimeError("ctransform_cuda_py not build - "
                           "run cmake --build build first")
    sys.path.insert(0, str(pathlib.Path(matches[0]).parent))

    # for `import ctransform_cuda.jax`
    sys.path.insert(0, str(root / "python"))


_find_build_dir()
