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
