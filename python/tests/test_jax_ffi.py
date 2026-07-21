import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import ctransform_cuda.jax as ctj


# ---- 1D ----

def ctransform_ref_1d(X, Y, phi):
    return np.array([np.min(0.5 * (X - y) ** 2 - phi) for y in Y])


def test_jax_ffi_1d_matches_reference():
    rng = np.random.default_rng(0)
    X = jnp.array(rng.uniform(0, 1, 100))
    Y = jnp.array(rng.uniform(0, 1, 80))
    phi = jnp.array(rng.uniform(-1, 1, 100))

    got = ctj.ctransform_1d(X, Y, phi)
    want = ctransform_ref_1d(np.array(X), np.array(Y), np.array(phi))
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_1d_inside_jit():
    rng = np.random.default_rng(11)
    X = jnp.array(rng.uniform(0, 1, 50))
    Y = jnp.array(rng.uniform(0, 1, 40))
    phi = jnp.array(rng.uniform(-1, 1, 50))

    jitted = jax.jit(ctj.ctransform_1d)
    got = jitted(X, Y, phi)
    want = ctransform_ref_1d(np.array(X), np.array(Y), np.array(phi))
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


# ---- 2D ----

def ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi):
    c0 = 0.5 * (Xaxis0[:, None] - Yaxis0[None, :]) ** 2
    c1 = 0.5 * (Xaxis1[:, None] - Yaxis1[None, :]) ** 2
    cost = c0[:, None, :, None] + c1[None, :, None, :]
    return (cost - phi[:, :, None, None]).min(axis=(0, 1))


def test_jax_ffi_2d_matches_reference():
    rng = np.random.default_rng(2)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))

    got = ctj.ctransform_2d(
        jnp.array(Xaxis0), jnp.array(Xaxis1),
        jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi)
    )
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_2d_separable_matches_reference():
    rng = np.random.default_rng(2)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))

    got = ctj.ctransform_2d_separable(
        jnp.array(Xaxis0), jnp.array(Xaxis1),
        jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi)
    )
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    assert got.shape == (5, 4)
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_2d_inside_jit():
    rng = np.random.default_rng(3)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))

    jitted = jax.jit(ctj.ctransform_2d)
    got = jitted(
        jnp.array(Xaxis0), jnp.array(Xaxis1),
        jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi)
    )
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_2d_separable_inside_jit():
    rng = np.random.default_rng(3)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))

    jitted = jax.jit(ctj.ctransform_2d_separable)
    got = jitted(
        jnp.array(Xaxis0), jnp.array(Xaxis1),
        jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi)
    )
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_2d_separable_matches_naive():
    rng = np.random.default_rng(6)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 64), rng.uniform(0, 1, 48)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 40), rng.uniform(0, 1, 32)
    phi = rng.uniform(-1, 1, (64, 48))

    args = (jnp.array(Xaxis0), jnp.array(Xaxis1),
            jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi))

    sep = ctj.ctransform_2d_separable(*args)
    naive = ctj.ctransform_2d(*args)
    np.testing.assert_allclose(np.array(sep), np.array(naive), atol=1e-12)


def test_jax_ffi_2d_separable_scratch_larger_than_output():
    # nx0 > ny0, so the scratch buffer (nx0*ny1) is larger than the output
    # (ny0*ny1). Guards the dOut/dScratchG argument order in the handler: if
    # they are ever swapped, pass 1 writes nx0*ny1 doubles into a ny0*ny1
    # allocation. Run under compute-sanitizer to see it as a fault rather
    # than as wrong values.
    rng = np.random.default_rng(7)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 9), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 5)
    phi = rng.uniform(-1, 1, (9, 7))

    got = ctj.ctransform_2d_separable(
        jnp.array(Xaxis0), jnp.array(Xaxis1),
        jnp.array(Yaxis0), jnp.array(Yaxis1), jnp.array(phi)
    )
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    assert got.shape == (3, 5)
    np.testing.assert_allclose(np.array(got), want, atol=1e-12)


def test_jax_ffi_2d_vmap_matches_loop():
    rng = np.random.default_rng(8)
    B = 4
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi_stack = rng.uniform(-1, 1, (B, 6, 7))

    X0, X1 = jnp.array(Xaxis0), jnp.array(Xaxis1)
    Y0, Y1 = jnp.array(Yaxis0), jnp.array(Yaxis1)

    batched = jax.vmap(
        lambda p: ctj.ctransform_2d(X0, X1, Y0, Y1, p), in_axes=0, out_axes=0
    )(jnp.array(phi_stack))

    want = np.stack([
        ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi_stack[b])
        for b in range(B)
    ])
    assert batched.shape == (B, 5, 4)
    np.testing.assert_allclose(np.array(batched), want, atol=1e-12)


def test_jax_ffi_2d_separable_vmap_inside_jit():
    rng = np.random.default_rng(9)
    B = 3
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 5), rng.uniform(0, 1, 4)
    phi_stack = rng.uniform(-1, 1, (B, 6, 7))

    X0, X1 = jnp.array(Xaxis0), jnp.array(Xaxis1)
    Y0, Y1 = jnp.array(Yaxis0), jnp.array(Yaxis1)

    fn = jax.jit(jax.vmap(
        lambda p: ctj.ctransform_2d_separable(X0, X1, Y0, Y1, p),
        in_axes=0, out_axes=0,
    ))
    batched = fn(jnp.array(phi_stack))

    want = np.stack([
        ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi_stack[b])
        for b in range(B)
    ])
    np.testing.assert_allclose(np.array(batched), want, atol=1e-12)
