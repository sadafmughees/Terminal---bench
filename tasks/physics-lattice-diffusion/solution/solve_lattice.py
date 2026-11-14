import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

RHO, C, K0, ALPHA = 2330.0, 700.0, 150.0, 0.003
T0, TAMB = 300.0, 300.0
L = 0.01
ETA, GAMMA, Q0 = 5.0e5, 8.0, 5.0e8
EPS, SIG = 0.85, 5.670374e-8


def _kappa_jax(T, alpha):
    return K0 / (1.0 + alpha * (T - T0))


_dkappa_elem = jax.jit(jax.vmap(jax.grad(_kappa_jax, argnums=0), in_axes=(0, None)))


def _mms_point_source(x, y, alpha, gamma, eta, q0):
    A = 50.0

    def T_fn(xv, yv):
        return T0 + A * jnp.sin(jnp.pi * xv / L) * jnp.sin(jnp.pi * yv / L)

    def flux_x(xv, yv):
        return _kappa_jax(T_fn(xv, yv), alpha) * jax.grad(T_fn, argnums=0)(xv, yv)

    def flux_y(xv, yv):
        return _kappa_jax(T_fn(xv, yv), alpha) * jax.grad(T_fn, argnums=1)(xv, yv)

    div = jax.grad(flux_x, argnums=0)(x, y) + jax.grad(flux_y, argnums=1)(x, y)
    Tv = T_fn(x, y)
    Q = (
        q0
        * jnp.exp(-eta * ((x - L / 2.0) ** 2 + (y - L / 2.0) ** 2))
        * jnp.exp(gamma * (Tv - T0) / T0)
    )
    return -(div + Q)


_mms_source_vec = jax.jit(
    jax.vmap(_mms_point_source, in_axes=(0, 0, None, None, None, None))
)


@jax.jit(static_argnums=(2, 6))
def _residual_jax(T_flat, q0, n, alpha, gamma, shape_flat, mms, src_flat):
    dx = dy = L / n
    Tm = T_flat.reshape((n, n))
    k = _kappa_jax(Tm, alpha)
    R = jnp.zeros((n, n), dtype=jnp.float64)

    kf_x = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    fx = kf_x * (Tm[:, 1:] - Tm[:, :-1]) / dx
    R = R.at[:, :-1].add(fx / dx)
    R = R.at[:, 1:].add(-fx / dx)

    kf_y = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    fy = kf_y * (Tm[1:, :] - Tm[:-1, :]) / dy
    R = R.at[:-1, :].add(fy / dy)
    R = R.at[1:, :].add(-fy / dy)

    kb = _kappa_jax(T0, alpha)
    if mms:
        for sl, h in (
            ((slice(None), 0), dx),
            ((slice(None), -1), dx),
            ((0, slice(None)), dy),
            ((-1, slice(None)), dy),
        ):
            kc = k[sl]
            kf = 2.0 * kc * kb / (kc + kb)
            R = R.at[sl].add(kf * (T0 - Tm[sl]) / (h / 2.0) / h)
    else:
        kc_bot = k[0, :]
        kf_bot = 2.0 * kc_bot * kb / (kc_bot + kb)
        R = R.at[0, :].add(kf_bot * (T0 - Tm[0, :]) / (dy / 2.0) / dy)
        R = R.at[-1, :].add(-EPS * SIG * (Tm[-1, :] ** 4 - TAMB ** 4) / dy)

    Q = q0 * shape_flat.reshape((n, n)) * jnp.exp(gamma * (Tm - T0) / T0)
    R = (R + Q).ravel()
    if mms:
        R = R + src_flat
    return R


class Problem:
    def __init__(self, n, alpha, gamma, eta=ETA, mms=False):
        self.n, self.alpha, self.gamma, self.eta = n, alpha, gamma, eta
        self.dx = self.dy = L / n
        xc = (np.arange(n) + 0.5) * self.dx
        X, Y = np.meshgrid(xc, xc)
        self.X, self.Y = X, Y
        self.shape = np.exp(-eta * ((X - L / 2.0) ** 2 + (Y - L / 2.0) ** 2)).ravel()
        self.shape_j = jnp.asarray(self.shape)
        self.mms = mms
        if mms:
            A = 50.0
            self.A = A
            s = A * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
            self.Tex = (T0 + s).ravel()
            src_j = _mms_source_vec(
                jnp.asarray(X.ravel()),
                jnp.asarray(Y.ravel()),
                float(alpha),
                float(gamma),
                float(eta),
                float(Q0),
            )
            self.src = np.asarray(src_j)
            self.src_j = src_j
        else:
            self.src = np.zeros(n * n)
            self.src_j = jnp.zeros(n * n)

    def kappa(self, T):
        return np.asarray(_kappa_jax(jnp.asarray(T), self.alpha))

    def dkappa(self, T):
        arr = np.asarray(T)
        return np.asarray(_dkappa_elem(jnp.asarray(arr.ravel()), self.alpha)).reshape(arr.shape)

    def residual(self, T, q0, jac=True):
        n, dx, dy = self.n, self.dx, self.dy
        R = np.asarray(
            _residual_jax(
                jnp.asarray(T),
                float(q0),
                int(n),
                float(self.alpha),
                float(self.gamma),
                self.shape_j,
                bool(self.mms),
                self.src_j,
            )
        )
        if not jac:
            return R, None, None

        Tm = T.reshape(n, n)
        k = self.kappa(Tm)
        dk = self.dkappa(Tm)
        idx = np.arange(n * n).reshape(n, n)

        def add_face(Ta, Tb, ka, kb, dka, dkb, h):
            kf = 2.0 * ka * kb / (ka + kb)
            dkf_a = 2.0 * kb * kb / (ka + kb) ** 2 * dka
            dkf_b = 2.0 * ka * ka / (ka + kb) ** 2 * dkb
            dfa = (dkf_a * (Tb - Ta) - kf) / h
            dfb = (dkf_b * (Tb - Ta) + kf) / h
            return dfa, dfb

        dfa, dfb = add_face(Tm[:, :-1], Tm[:, 1:], k[:, :-1], k[:, 1:], dk[:, :-1], dk[:, 1:], dx)
        dga, dgb = add_face(Tm[:-1, :], Tm[1:, :], k[:-1, :], k[1:, :], dk[:-1, :], dk[1:, :], dy)

        N = n * n
        I = idx
        rows, cols, vals = [], [], []
        a = I[:, :-1].ravel()
        b = I[:, 1:].ravel()
        rows += [a, a, b, b]
        cols += [a, b, a, b]
        vals += [dfa.ravel() / dx, dfb.ravel() / dx, -dfa.ravel() / dx, -dfb.ravel() / dx]

        a = I[:-1, :].ravel()
        b = I[1:, :].ravel()
        rows += [a, a, b, b]
        cols += [a, b, a, b]
        vals += [dga.ravel() / dy, dgb.ravel() / dy, -dga.ravel() / dy, -dgb.ravel() / dy]

        d = np.zeros((n, n))
        kb = self.kappa(T0)
        if self.mms:
            for sl_cells, h, vol in (
                (np.s_[:, 0], dx, dx),
                (np.s_[:, -1], dx, dx),
                (np.s_[0, :], dy, dy),
                (np.s_[-1, :], dy, dy),
            ):
                kc = k[sl_cells]
                dkc = dk[sl_cells]
                kf = 2.0 * kc * kb / (kc + kb)
                dkf = 2.0 * kb * kb / (kc + kb) ** 2 * dkc
                d[sl_cells] += (dkf * (T0 - Tm[sl_cells]) - kf) / (h / 2.0) / vol
        else:
            kc = k[0, :]
            dkc = dk[0, :]
            kf = 2.0 * kc * kb / (kc + kb)
            dkf = 2.0 * kb * kb / (kc + kb) ** 2 * dkc
            d[0, :] += (dkf * (T0 - Tm[0, :]) - kf) / (dy / 2.0) / dy
            d[-1, :] -= 4.0 * EPS * SIG * Tm[-1, :] ** 3 / dy

        Q = q0 * self.shape.reshape(n, n) * np.exp(self.gamma * (Tm - T0) / T0)
        d += Q * self.gamma / T0
        rows.append(I.ravel())
        cols.append(I.ravel())
        vals.append(d.ravel())

        J = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N),
        )
        dRdq = (self.shape.reshape(n, n) * np.exp(self.gamma * (Tm - T0) / T0)).ravel()
        return R, J, dRdq


def newton(P, T, q0, tol=1e-8, maxit=60):
    scale = max(1.0, np.linalg.norm(P.shape * q0))
    for it in range(maxit):
        R, J, _ = P.residual(T, q0)
        res = np.linalg.norm(R) / scale
        if res < tol:
            return T, True, it
        dT = spla.spsolve(J, -R)
        step = 1.0
        while step > 1e-4:
            Tn = T + step * dT
            Rn, _, _ = P.residual(Tn, q0, jac=False)
            if np.linalg.norm(Rn) / scale < res * (1.0 - 1e-4 * step) or np.linalg.norm(Rn) / scale < tol:
                break
            step *= 0.5
        T = Tn
    return T, False, maxit


TS, QS = 50.0, 6.0e8


def continuation(P, ds=0.05, max_steps=400):
    N = P.n * P.n
    T = np.full(N, T0)
    q = 0.0
    qn_ = 0.02 * QS
    T1, ok, _ = newton(P, T, qn_)
    pts = [(q, T.copy()), (qn_, T1.copy())]
    tT = (T1 - T) / TS
    tq = (qn_ - q) / QS
    nrm = np.sqrt(tT @ tT / N + tq ** 2)
    tT, tq = tT / nrm, tq / nrm
    cur_T, cur_q = T1.copy(), qn_
    for _ in range(max_steps):
        Tn = cur_T + ds * TS * tT
        qn = cur_q + ds * QS * tq
        done = False
        for _ in range(30):
            R, J, dRdq = P.residual(Tn, qn)
            g = (tT @ (Tn - cur_T) / TS) / N + tq * (qn - cur_q) / QS - ds
            lu = spla.splu(J)
            a = lu.solve(-R)
            b = lu.solve(-dRdq)
            den = (tT @ b / TS) / N + tq / QS
            dq = (-g - (tT @ a / TS) / N) / den
            dT = a + b * dq
            Tn = Tn + dT
            qn = qn + dq
            if np.linalg.norm(dT) / np.sqrt(N) < 1e-9 and abs(dq) < 1e-9 * QS:
                done = True
                break
        if not done:
            ds *= 0.5
            if ds < 1e-5:
                raise RuntimeError("continuation failed")
            continue
        R, J, dRdq = P.residual(Tn, qn)
        v = spla.splu(J).solve(-dRdq)
        tT_new = v / TS
        tq_new = 1.0 / QS
        nn = np.sqrt(tT_new @ tT_new / N + tq_new ** 2)
        tT_new, tq_new = tT_new / nn, tq_new / nn
        if (tT_new @ tT) / N + tq_new * tq < 0:
            tT_new, tq_new = -tT_new, -tq_new
        tT, tq = tT_new, tq_new
        cur_T, cur_q = Tn, qn
        pts.append((qn, Tn.copy()))
        if len(pts) > 3 and pts[-1][0] < pts[-2][0]:
            break
    return pts


def fold_from_path(pts):
    q = np.array([p[0] for p in pts])
    Tm = np.array([p[1].max() for p in pts])
    k = int(np.argmax(q))
    lo, hi = max(k - 2, 0), min(k + 3, len(q))
    c = np.polyfit(Tm[lo:hi], q[lo:hi], 2)
    Tc = -c[1] / (2.0 * c[0])
    Qc = np.polyval(c, Tc)
    return Qc, Tc


def stability_index(P, T, q0):
    _, J, _ = P.residual(T, q0)
    A = J / (RHO * C)
    vals = spla.eigs(A.tocsc(), k=6, sigma=0.0, which="LM", return_eigenvectors=False)
    return float(np.max(vals.real))


def mms_slope():
    errs, hs = [], []
    for n in (25, 50, 100):
        P = Problem(n, ALPHA, GAMMA, mms=True)
        T = np.full(n * n, T0 + 10.0)
        T, ok, _ = newton(P, T, Q0, tol=1e-9)
        if not ok:
            raise RuntimeError(f"MMS Newton failed at n={n}")
        e = np.sqrt(np.mean((T - P.Tex) ** 2))
        errs.append(e)
        hs.append(L / n)
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def solve(alpha, gamma, q0, n=100, eta=ETA):
    P = Problem(n, alpha, gamma, eta)
    pts = continuation(P)
    Qc, Tc = fold_from_path(pts)
    if q0 >= Qc:
        raise ValueError("q0 must lie below Q_crit")
    below = [p for p in pts if p[0] <= q0]
    T, ok, _ = newton(P, below[-1][1], q0, tol=1e-10)
    if not ok:
        raise RuntimeError("Newton failed at q0")
    return P, T, Qc, Tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--q0", type=float, default=Q0)
    ap.add_argument("--out", default="/app/solution_metrics.json")
    ap.add_argument("--skip-mms", action="store_true")
    a = ap.parse_args()
    P, T, Qc, Tc = solve(a.alpha, a.gamma, a.q0)
    out = {
        "T_field": T.reshape(P.n, P.n).tolist(),
        "Q_crit": float(Qc),
        "T_crit": float(Tc),
        "S": stability_index(P, T, a.q0),
    }
    if not a.skip_mms:
        out["mms_slope"] = mms_slope()
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
