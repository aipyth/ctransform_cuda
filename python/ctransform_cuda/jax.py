import jax
from . import _ctransform_ffi

jax.ffi.register_ffi_target(
        "ctransform_1d_f64",
        _ctransform_ffi.ctransform_1d_f64(),
        platform="CUDA",
        )
jax.ffi.register_ffi_target(
        "ctransform_2d_f64",
        _ctransform_ffi.ctransform_2d_f64(),
        platform="CUDA",
        )
jax.ffi.register_ffi_target(
        "ctransform_2d_separable_f64",
        _ctransform_ffi.ctransform_2d_separable_f64(),
        platform="CUDA",
        )
jax.ffi.register_ffi_target(
        "ctransform_3d_f64",
        _ctransform_ffi.ctransform_3d_f64(),
        platform="CUDA",
        )
jax.ffi.register_ffi_target(
        "ctransform_3d_separable_f64",
        _ctransform_ffi.ctransform_3d_separable_f64(),
        platform="CUDA",
        )


def ctransform_1d(X, Y, phi):
    out_type = jax.ShapeDtypeStruct(Y.shape, X.dtype)
    return jax.ffi.ffi_call(
            "ctransform_1d_f64", out_type, vmap_method="sequential"
            )(X, Y, phi)


def ctransform_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi):
    out_type = jax.ShapeDtypeStruct((Yaxis0.shape[0], Yaxis1.shape[0]), Xaxis0.dtype)
    return jax.ffi.ffi_call(
            "ctransform_2d_f64", out_type, vmap_method="sequential"
            )(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)


def ctransform_2d_separable(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi):
    out_type = jax.ShapeDtypeStruct((Yaxis0.shape[0], Yaxis1.shape[0]), Xaxis0.dtype)
    return jax.ffi.ffi_call(
            "ctransform_2d_separable_f64", out_type,
            vmap_method="sequential"
            )(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)


def ctransform_3d(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi):
    """3D quadratic c-transform, naive kernel.

    out[i, j, k] = min over source points x of 0.5 * |x - y|^2 - phi(x),
    with y = (Yaxis0[i], Yaxis1[j], Yaxis2[k]). Cost grows as n^6: for testing
    and small grids; use ctransform_3d_separable otherwise.
    """
    out_type = jax.ShapeDtypeStruct(
        (Yaxis0.shape[0], Yaxis1.shape[0], Yaxis2.shape[0]), phi.dtype)
    return jax.ffi.ffi_call(
            "ctransform_3d_f64", out_type, vmap_method="sequential"
            )(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi)

def ctransform_3d_separable(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi):
    """3D quadratic c-transform, separable kernel: same result as ctransform_3d, cost ~3 n^4.

    For the reverse direction (a function on the Y grid transformed back onto the
    X grid), call it with the X and Y axes swapped.
    """
    out_type = jax.ShapeDtypeStruct(
        (Yaxis0.shape[0], Yaxis1.shape[0], Yaxis2.shape[0]), phi.dtype)
    return jax.ffi.ffi_call(
            "ctransform_3d_separable_f64", out_type, vmap_method="sequential"
            )(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi)
