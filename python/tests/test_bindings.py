import numpy as np
import pytest
import ctransform_cuda_py as ct


# ---- 1D ----

def ctransform_ref_1d(X, Y, phi):
    return np.array([np.min(0.5 * (X - y) ** 2 - phi) for y in Y])


def test_1d_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, 100)
    Y = rng.uniform(0, 1, 80)
    phi = rng.uniform(-1, 1, 100)
    np.testing.assert_allclose(ct.ctransform_1d(X, Y, phi), ctransform_ref_1d(X, Y, phi), atol=1e-12)


def test_1d_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ct.ctransform_1d(np.zeros(3), np.zeros(2), np.zeros(4))  # phi.shape != X.shape


def test_1d_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        ct.ctransform_1d(np.zeros((3, 1)), np.zeros(2), np.zeros(3))


# ---- 2D ----

def ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi):
    # cost[ix0,ix1,iy0,iy1] = 0.5*((Xaxis0[ix0]-Yaxis0[iy0])**2 + (Xaxis1[ix1]-Yaxis1[iy1])**2)
    c0 = 0.5 * (Xaxis0[:, None] - Yaxis0[None, :]) ** 2     # (nx0, ny0)
    c1 = 0.5 * (Xaxis1[:, None] - Yaxis1[None, :]) ** 2     # (nx1, ny1)
    cost = c0[:, None, :, None] + c1[None, :, None, :]      # (nx0, nx1, ny0, ny1)
    return (cost - phi[:, :, None, None]).min(axis=(0, 1))


def test_2d_matches_reference():
    rng = np.random.default_rng(0)
    Xaxis0 = rng.uniform(0, 1, 6)
    Xaxis1 = rng.uniform(0, 1, 7)
    Yaxis0 = rng.uniform(0, 1, 5)
    Yaxis1 = rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))
    got = ct.ctransform_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    want = ctransform_ref_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
    assert got.shape == (5, 4)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_2d_rejects_phi_shape_mismatch():
    with pytest.raises(ValueError):
        ct.ctransform_2d(
                np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2),
                np.zeros((4, 4)),   # wrong: should be (3, 3)
                )

def test_2d_rejects_wrong_phi_ndim():
    with pytest.raises(ValueError):
        ct.ctransform_2d(np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2), np.zeros(9))


def test_2d_separable_matches_naive():
    rng = np.random.default_rng(1)
    Xaxis0, Xaxis1 = rng.uniform(0, 1, 6), rng.uniform(0, 1, 7)
    Yaxis0, Yaxis1 = rng.uniform(0, 1, 4), rng.uniform(0, 1, 4)
    phi = rng.uniform(-1, 1, (6, 7))
    naive = ct.ctransform_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi, method="naive")
    separable = ct.ctransform_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi, method="separable")
    np.testing.assert_allclose(naive, separable, atol=1e-12)


def test_2d_rejects_unknown_method():
    with pytest.raises(ValueError):
        ct.ctransform_2d(np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2), np.zeros((3, 3)), method="bogus")
