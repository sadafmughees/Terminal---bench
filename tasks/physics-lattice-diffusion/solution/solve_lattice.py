import argparse
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RHO, C, K0, ALPHA = 2330.0, 700.0, 150.0, 0.003
T0, TAMB = 300.0, 300.0
L = 0.01
ETA, GAMMA, Q0 = 5.0e5, 8.0, 5.0e8
EPS, SIG = 0.85, 5.670374e-8


class Problem:
    def __init__(self, n, alpha, gamma, eta=ETA, mms=False):
        self.n, self.alpha, self.gamma, self.eta = n, alpha, gamma, eta
        self.dx = self.dy = L / n
        xc = (np.arange(n) + 0.5) * self.dx
        X, Y = np.meshgrid(xc, xc)  # X[i,j]=x_j, Y[i,j]=y_i
        self.X, self.Y = X, Y
        self.shape = np.exp(-eta * ((X - L / 2) ** 2 + (Y - L / 2) ** 2)).ravel()
        self.mms = mms
        if mms:
            A = 50.0
            self.A = A
            s = A * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
            self.Tex = (T0 + s).ravel()
            # exact fields on boundary (zero rise on all four edges)
            self.src = self._mms_source(X, Y)

    def kappa(self, T):
        return K0 / (1.0 + self.alpha * (T - T0))

    def dkappa(self, T):
        return -K0 * self.alpha / (1.0 + self.alpha * (T - T0)) ** 2

    def _mms_source(self, X, Y, q0=Q0):
        A = self.A
        sx, sy = np.sin(np.pi * X / L), np.sin(np.pi * Y / L)
        cx, cy = np.cos(np.pi * X / L), np.cos(np.pi * Y / L)
        T = T0 + A * sx * sy
        lap = -2 * (np.pi / L) ** 2 * A * sx * sy
        gx, gy = A * np.pi / L * cx * sy, A * np.pi / L * sx * cy
        div = self.dkappa(T) * (gx ** 2 + gy ** 2) + self.kappa(T) * lap
        Q = q0 * np.exp(-self.eta * ((X - L / 2) ** 2 + (Y - L / 2) ** 2)) * np.exp(self.gamma * (T - T0) / T0)
        return (-(div + Q)).ravel()

    # residual R(T,q0) = div(kappa grad T) + Q (+ MMS source), per unit volume
    def residual(self, T, q0, jac=True):
        n, dx, dy = self.n, self.dx, self.dy
        Tm = T.reshape(n, n)
        k = self.kappa(Tm)
        dk = self.dkappa(Tm)
        R = np.zeros((n, n))
        idx = np.arange(n * n).reshape(n, n)

        def add_face(Ta, Tb, ka, kb, dka, dkb, ia, ib, h):
            # flux from b into a through a face: kf (Tb-Ta)/h, kf = harmonic mean
            kf = 2 * ka * kb / (ka + kb)
            f = kf * (Tb - Ta) / h
            if jac:
                dkf_a = 2 * kb * kb / (ka + kb) ** 2 * dka
                dkf_b = 2 * ka * ka / (ka + kb) ** 2 * dkb
                dfa = (dkf_a * (Tb - Ta) - kf) / h
                dfb = (dkf_b * (Tb - Ta) + kf) / h
                return f, dfa, dfb
            return f, None, None

        # x faces (between j and j+1); lateral outer faces adiabatic
        f, dfa, dfb = add_face(Tm[:, :-1], Tm[:, 1:], k[:, :-1], k[:, 1:], dk[:, :-1], dk[:, 1:], None, None, dx)
        R[:, :-1] += f / dx
        R[:, 1:] -= f / dx
        # y faces between i and i+1
        g, dga, dgb = add_face(Tm[:-1, :], Tm[1:, :], k[:-1, :], k[1:, :], dk[:-1, :], dk[1:, :], None, None, dy)
        R[:-1, :] += g / dy
        R[1:, :] -= g / dy

        # boundaries
        if self.mms:
            # Dirichlet from exact on all four edges (exact value T0 on boundary)
            Tb = T0
            hh = dx / 2
            # left/right
            for sl_cells, h, vol in ((np.s_[:, 0], dx, dx), (np.s_[:, -1], dx, dx), (np.s_[0, :], dy, dy), (np.s_[-1, :], dy, dy)):
                kc = k[sl_cells]
                kb = self.kappa(Tb)
                kf = 2 * kc * kb / (kc + kb)
                R[sl_cells] += kf * (Tb - Tm[sl_cells]) / (h / 2) / vol
        else:
            # bottom Dirichlet T0 at y=0 (row 0)
            kc = k[0, :]
            kb = self.kappa(T0)
            kf = 2 * kc * kb / (kc + kb)
            R[0, :] += kf * (T0 - Tm[0, :]) / (dy / 2) / dy
            # top radiative at y=L (row -1): surface temp taken at the cell centre
            R[-1, :] -= EPS * SIG * (Tm[-1, :] ** 4 - TAMB ** 4) / dy
        Q = q0 * self.shape.reshape(n, n) * np.exp(self.gamma * (Tm - T0) / T0)
        R += Q
        R = R.ravel()
        if self.mms:
            R = R + self.src
        if not jac:
            return R, None, None

        # Jacobian
        N = n * n
        I = idx
        rows, cols, vals = [], [], []
        # x faces
        a = I[:, :-1].ravel(); b = I[:, 1:].ravel()
        rows += [a, a, b, b]; cols += [a, b, a, b]
        vals += [dfa.ravel() / dx, dfb.ravel() / dx, -dfa.ravel() / dx, -dfb.ravel() / dx]
        a = I[:-1, :].ravel(); b = I[1:, :].ravel()
        rows += [a, a, b, b]; cols += [a, b, a, b]
        vals += [dga.ravel() / dy, dgb.ravel() / dy, -dga.ravel() / dy, -dgb.ravel() / dy]
        d = np.zeros((n, n))
        if self.mms:
            for sl_cells, h, vol in ((np.s_[:, 0], dx, dx), (np.s_[:, -1], dx, dx), (np.s_[0, :], dy, dy), (np.s_[-1, :], dy, dy)):
                kc = k[sl_cells]; dkc = dk[sl_cells]
                kb = self.kappa(T0)
                kf = 2 * kc * kb / (kc + kb)
                dkf = 2 * kb * kb / (kc + kb) ** 2 * dkc
                d[sl_cells] += (dkf * (T0 - Tm[sl_cells]) - kf) / (h / 2) / vol
        else:
            kc = k[0, :]; dkc = dk[0, :]
            kb = self.kappa(T0)
            kf = 2 * kc * kb / (kc + kb)
            dkf = 2 * kb * kb / (kc + kb) ** 2 * dkc
            d[0, :] += (dkf * (T0 - Tm[0, :]) - kf) / (dy / 2) / dy
            d[-1, :] -= 4 * EPS * SIG * Tm[-1, :] ** 3 / dy
        dQdT = Q * self.gamma / T0
        d += dQdT
        rows.append(I.ravel()); cols.append(I.ravel()); vals.append(d.ravel())
        J = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
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
            if np.linalg.norm(Rn) / scale < res * (1 - 1e-4 * step) or np.linalg.norm(Rn) / scale < tol:
                break
            step *= 0.5
        T = Tn
    return T, False, maxit



TS, QS = 50.0, 6.0e8   # scaling of temperature rise and source in the arclength norm


def continuation(P, ds=0.05, max_steps=400):
    """Pseudo-arclength continuation of the lower branch in Q0 until past the fold."""
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
        for it in range(30):
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
    # parabola q(Tmax) through points around the maximum
    c = np.polyfit(Tm[lo:hi], q[lo:hi], 2)
    Tc = -c[1] / (2 * c[0])
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
        T, ok, _ = newton(P, T, Q0, tol=1e-11)
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
    # lower-branch state at q0: Newton from the nearest earlier continuation point
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
    ap.add_argument("--out", default="output.json")
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
