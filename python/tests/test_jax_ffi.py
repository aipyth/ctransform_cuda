import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
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


# ---- 3D ----

def ctransform_ref_3d(Xaxes, Yaxes, phi):
    """Brute-force NumPy reference: min over all source points, for every target point."""
    c = [0.5 * (x[:, None] - y[None, :]) ** 2 for x, y in zip(Xaxes, Yaxes)]
    cost = (c[0][:, None, None, :, None, None]
            + c[1][None, :, None, None, :, None]
            + c[2][None, None, :, None, None, :])
    return (cost - phi[:, :, :, None, None, None]).min(axis=(0, 1, 2))


def random_problem_3d(seed, nx, ny):
    """Random axes in [0, 1], phi in [-1, 1]; returned as NumPy arrays."""
    rng = np.random.default_rng(seed)
    Xaxes = [rng.uniform(0, 1, n) for n in nx]
    Yaxes = [rng.uniform(0, 1, n) for n in ny]
    phi = rng.uniform(-1, 1, nx)
    return Xaxes, Yaxes, phi


def call_3d(fn, Xaxes, Yaxes, phi):
    return fn(*map(jnp.array, Xaxes), *map(jnp.array, Yaxes), jnp.array(phi))


KERNELS_3D = [
    pytest.param(ctj.ctransform_3d, id="naive"),
    pytest.param(ctj.ctransform_3d_separable, id="separable"),
]


@pytest.mark.parametrize("fn", KERNELS_3D)
def test_jax_ffi_3d_matches_reference(fn):
    # All six sizes differ, so a swapped axis or a wrong stride changes the result.
    Xaxes, Yaxes, phi = random_problem_3d(20, nx=(5, 6, 7), ny=(4, 8, 3))
    got = call_3d(fn, Xaxes, Yaxes, phi)
    assert got.shape == (4, 8, 3)
    np.testing.assert_allclose(np.array(got), ctransform_ref_3d(Xaxes, Yaxes, phi), atol=1e-12)


@pytest.mark.parametrize("fn", KERNELS_3D)
def test_jax_ffi_3d_inside_jit(fn):
    Xaxes, Yaxes, phi = random_problem_3d(21, nx=(4, 5, 6), ny=(6, 3, 5))
    got = call_3d(jax.jit(fn), Xaxes, Yaxes, phi)
    np.testing.assert_allclose(np.array(got), ctransform_ref_3d(Xaxes, Yaxes, phi), atol=1e-12)


def test_jax_ffi_3d_separable_matches_naive():
    # Larger than the NumPy reference can handle comfortably; naive is the reference here.
    Xaxes, Yaxes, phi = random_problem_3d(22, nx=(20, 18, 22), ny=(16, 24, 14))
    sep = call_3d(ctj.ctransform_3d_separable, Xaxes, Yaxes, phi)
    naive = call_3d(ctj.ctransform_3d, Xaxes, Yaxes, phi)
    np.testing.assert_allclose(np.array(sep), np.array(naive), atol=1e-12)


def test_jax_ffi_3d_separable_back_and_forth_in_fori_loop():
    # The solver pattern: alternate X -> Y and Y -> X transforms inside a traced loop.
    # For any phi, phi^cc >= phi, and after one round trip the pair is a fixed point:
    # the next round trip reproduces the same values.
    Xaxes, Yaxes, phi = random_problem_3d(23, nx=(6, 5, 7), ny=(5, 7, 4))
    X = tuple(map(jnp.array, Xaxes))
    Y = tuple(map(jnp.array, Yaxes))

    def round_trip(_, p):
        psi = ctj.ctransform_3d_separable(*X, *Y, p)      # function on Y
        return ctj.ctransform_3d_separable(*Y, *X, psi)   # back onto X

    phi_cc = jax.jit(lambda p: jax.lax.fori_loop(0, 1, round_trip, p))(jnp.array(phi))
    phi_cccc = jax.jit(lambda p: jax.lax.fori_loop(0, 3, round_trip, p))(jnp.array(phi))

    assert np.all(np.array(phi_cc) >= phi - 1e-12)
    np.testing.assert_allclose(np.array(phi_cccc), np.array(phi_cc), atol=1e-12)


def test_jax_ffi_3d_separable_vmap():
    # vmap_method="sequential": one kernel call per batch entry, each with its own scratch.
    Xaxes, Yaxes, _ = random_problem_3d(24, nx=(4, 5, 3), ny=(3, 4, 6))
    phis = np.random.default_rng(25).uniform(-1, 1, (3, 4, 5, 3))
    X = tuple(map(jnp.array, Xaxes))
    Y = tuple(map(jnp.array, Yaxes))

    batched = jax.vmap(lambda p: ctj.ctransform_3d_separable(*X, *Y, p))(jnp.array(phis))
    want = np.stack([ctransform_ref_3d(Xaxes, Yaxes, p) for p in phis])
    assert batched.shape == (3, 3, 4, 6)
    np.testing.assert_allclose(np.array(batched), want, atol=1e-12)


@pytest.mark.parametrize("fn", KERNELS_3D)
def test_jax_ffi_3d_rejects_wrong_phi_shape(fn):
    # The C++ handler checks phi against the axis lengths before launching anything.
    Xaxes, Yaxes, phi = random_problem_3d(26, nx=(4, 5, 6), ny=(3, 3, 3))
    with pytest.raises(jax.errors.JaxRuntimeError, match="phi must have shape"):
        call_3d(fn, Xaxes, Yaxes, phi.transpose(2, 1, 0)).block_until_ready()   # (6, 5, 4)


@pytest.mark.parametrize("fn", KERNELS_3D)
def test_jax_ffi_3d_rejects_float32(fn):
    # The handler is registered for float64 buffers only; XLA refuses anything else.
    Xaxes, Yaxes, phi = random_problem_3d(27, nx=(2, 2, 2), ny=(2, 2, 2))
    with pytest.raises(jax.errors.JaxRuntimeError, match="INVALID_ARGUMENT"):
        fn(*(jnp.array(a, dtype=jnp.float32) for a in (*Xaxes, *Yaxes, phi))).block_until_ready()
