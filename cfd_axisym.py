"""
2D-axisymmetrische RANS-solver (r, z) met swirl voor een geroerde tank.

Onbekenden (MAC/staggered rooster, eindige volumes, conservatief):
    u  radiale snelheid    op r-vlakken
    w  axiale snelheid     op z-vlakken
    G  = r * v_theta       impulsmoment (celcentra)
    c  tracerconcentratie  (celcentra)

- Fractional-step projectie (Chorin), drukvergelijking via voorgefactoriseerde LU.
- Turbulentie: Prandtl-menglengte  nu_t = l_m^2 |S|,  l_m = min(kappa*d, L_MAX).
- Wanden: log-wet wandfuncties (effectieve wandviscositeit).
- Waaiers: blade-element "virtual blade" model (vlakke plaat onder 45 graden),
  tijdgemiddeld over de omtrek -> bronterm in theta- en z-impuls.
- Keerschotten: weerstandsterm op swirl in de wandzone (porous baffle model).
- Driepoot: weerstand op (u, v, w) in de bodemzone rond de as.
"""
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

import config as C

KAPPA = 0.41
E_LOG = 9.8
T_RAMP = 5.0     # s  soft start ramp of the rotor speed


# ----------------------------------------------------------------------------- grid
class Grid:
    def __init__(self):
        self.r0 = C.SHAFT_OD / 2
        self.R = C.T_DIAM / 2
        self.H = C.H_LIQ
        self.Nr = int(round((self.R - self.r0) / C.DX))
        self.Nz = int(round(self.H / C.DX))
        self.dr = (self.R - self.r0) / self.Nr
        self.dz = self.H / self.Nz
        self.rc = self.r0 + (np.arange(self.Nr) + 0.5) * self.dr
        self.rf = self.r0 + np.arange(self.Nr + 1) * self.dr
        self.zc = (np.arange(self.Nz) + 0.5) * self.dz
        self.zf = np.arange(self.Nz + 1) * self.dz
        RC, ZC = np.meshgrid(self.rc, self.zc, indexing="ij")
        self.RC, self.ZC = RC, ZC
        self.vol = 2 * np.pi * RC * self.dr * self.dz
        # wall distance for mixing length
        self.dwall = np.minimum.reduce([RC - self.r0, self.R - RC, ZC, self.H - ZC])
        # inner-wall velocity: shaft rotates above the bottom bearing
        self.vwall_in = np.where(self.zc > C.Z_BEARING, C.OMEGA * self.r0, 0.0)


def overlap(a0, a1, b0, b1):
    return np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None)


def build_sources(g, baffles):
    """Static weight fields for impellers, baffles and tripod."""
    h_band = C.BLADE_W * np.sin(np.radians(C.PITCH_DEG))  # axial height swept by blade
    imp_w = np.zeros((g.Nr, g.Nz))
    imp_id = -np.ones((g.Nr, g.Nz), dtype=np.int64)
    fr = overlap(g.rc - g.dr / 2, g.rc + g.dr / 2, C.R_HUB, C.R_IMP) / g.dr
    for k, zi in enumerate(C.Z_IMPELLERS):
        fz = overlap(g.zc - g.dz / 2, g.zc + g.dz / 2, zi - h_band / 2, zi + h_band / 2) / g.dz
        wk = np.outer(fr, fz)
        imp_w += wk
        imp_id[wk > 0] = k
    imp_w = imp_w / (2 * np.pi * g.RC * h_band)    # -> per unit volume normalisation
    baf = np.zeros((g.Nr, g.Nz))
    if baffles:
        fb = overlap(g.rc - g.dr / 2, g.rc + g.dr / 2, g.R - C.BAFFLE_W, g.R) / g.dr
        baf[:] = (C.N_BAFFLES * C.CD_BAFFLE / (4 * np.pi * g.rc) * fb)[:, None]
    # tripod: legs from bearing (r0, Z_BEARING) to floor at TRIPOD_R
    leg_len = np.hypot(C.TRIPOD_R - g.r0, C.Z_BEARING)
    v_reg = np.pi * (C.TRIPOD_R**2 - g.r0**2) * C.TRIPOD_H
    a_trip = C.TRIPOD_LEGS * C.CD_LEG * C.TRIPOD_LEG_D * leg_len / (2 * v_reg)
    trip = np.where((g.RC < C.TRIPOD_R) & (g.ZC < C.TRIPOD_H), a_trip, 0.0)
    return imp_w, imp_id, h_band, baf, trip


def build_poisson(g):
    Nr, Nz, dr, dz, rc, rf = g.Nr, g.Nz, g.dr, g.dz, g.rc, g.rf
    N = Nr * Nz
    rows, cols, vals = [], [], []
    idx = lambda i, j: i * Nz + j
    for i in range(Nr):
        for j in range(Nz):
            n = idx(i, j)
            d = 0.0
            if i < Nr - 1:
                a = rf[i + 1] / (rc[i] * dr * dr); rows.append(n); cols.append(idx(i + 1, j)); vals.append(a); d -= a
            if i > 0:
                a = rf[i] / (rc[i] * dr * dr); rows.append(n); cols.append(idx(i - 1, j)); vals.append(a); d -= a
            if j < Nz - 1:
                a = 1 / dz**2; rows.append(n); cols.append(idx(i, j + 1)); vals.append(a); d -= a
            if j > 0:
                a = 1 / dz**2; rows.append(n); cols.append(idx(i, j - 1)); vals.append(a); d -= a
            rows.append(n); cols.append(n); vals.append(d)
    A = sp.csc_matrix((vals, (rows, cols)), shape=(N, N)).tolil()
    A[0, :] = 0
    A[0, 0] = 1.0
    return spla.splu(A.tocsc())


# ----------------------------------------------------------------------------- kernels
@njit(cache=True)
def wall_nu(U, y, nu):
    if U < 1e-9:
        return nu
    ut = np.sqrt(nu * U / y)
    if ut * y / nu > 11.06:
        for _ in range(8):
            yp = max(ut * y / nu, 11.06)
            ut = KAPPA * U / np.log(E_LOG * yp)
    return max(ut * ut * y / U, nu)


@njit(cache=True)
def k_turb(u, w, G, rc, dr, dz, r0, vwin, dwall, lmax, nu, nut,
           nw_in, nw_out, nw_bot, nw_top):
    Nr, Nz = G.shape
    for i in range(Nr):
        for j in range(Nz):
            uc = 0.5 * (u[i, j] + u[i + 1, j])
            wc = 0.5 * (w[i, j] + w[i, j + 1])
            v = G[i, j] / rc[i]
            dudr = (u[i + 1, j] - u[i, j]) / dr
            dwdz = (w[i, j + 1] - w[i, j]) / dz
            # du/dz
            if j == 0:
                dudz = (0.5 * (u[i, 1] + u[i + 1, 1]) - uc) / dz
            elif j == Nz - 1:
                dudz = (uc - 0.5 * (u[i, j - 1] + u[i + 1, j - 1])) / dz
            else:
                dudz = (0.5 * (u[i, j + 1] + u[i + 1, j + 1]) - 0.5 * (u[i, j - 1] + u[i + 1, j - 1])) / (2 * dz)
            # dw/dr, d(v/r)/dr
            if i == 0:
                dwdr = (0.5 * (w[1, j] + w[1, j + 1]) - wc) / dr
                om_l = vwin[j] / r0
                om_r = G[1, j] / rc[1] ** 2
                domdr = (om_r - om_l) / (1.5 * dr)
            elif i == Nr - 1:
                dwdr = (wc - 0.5 * (w[i - 1, j] + w[i - 1, j + 1])) / dr
                om_l = G[i - 1, j] / rc[i - 1] ** 2
                domdr = (0.0 - om_l) / (1.5 * dr)
            else:
                dwdr = (0.5 * (w[i + 1, j] + w[i + 1, j + 1]) - 0.5 * (w[i - 1, j] + w[i - 1, j + 1])) / (2 * dr)
                domdr = (G[i + 1, j] / rc[i + 1] ** 2 - G[i - 1, j] / rc[i - 1] ** 2) / (2 * dr)
            if j == 0:
                dvdz = (G[i, 1] / rc[i] - 0.0) / (1.5 * dz)
            elif j == Nz - 1:
                dvdz = (0.0 - G[i, j - 1] / rc[i]) / (1.5 * dz)
            else:
                dvdz = (G[i, j + 1] - G[i, j - 1]) / rc[i] / (2 * dz)
            S2 = 2 * (dudr**2 + (uc / rc[i]) ** 2 + dwdz**2) + (dudz + dwdr) ** 2 \
                + (rc[i] * domdr) ** 2 + dvdz**2
            lm = min(KAPPA * dwall[i, j], lmax)
            nut[i, j] = nu + lm * lm * np.sqrt(S2)
    # wall functions
    y = 0.5 * dr
    for j in range(Nz):
        wc0 = 0.5 * (w[0, j] + w[0, j + 1])
        v0 = G[0, j] / rc[0] - vwin[j]
        nw_in[j] = wall_nu(np.sqrt(v0 * v0 + wc0 * wc0), y, nu)
        wcN = 0.5 * (w[Nr - 1, j] + w[Nr - 1, j + 1])
        vN = G[Nr - 1, j] / rc[Nr - 1]
        nw_out[j] = wall_nu(np.sqrt(vN * vN + wcN * wcN), y, nu)
    y = 0.5 * dz
    for i in range(Nr):
        uc0 = 0.5 * (u[i, 0] + u[i + 1, 0])
        v0 = G[i, 0] / rc[i]
        nw_bot[i] = wall_nu(np.sqrt(uc0 * uc0 + v0 * v0), y, nu)
        ucN = 0.5 * (u[i, Nz - 1] + u[i + 1, Nz - 1])
        vN = G[i, Nz - 1] / rc[i]
        nw_top[i] = wall_nu(np.sqrt(ucN * ucN + vN * vN), y, nu)


@njit(cache=True)
def blend(a, b, vel, gam):
    # donor-cell / central blend (Hirt), a = upstream side for vel>0
    s = 1.0 if vel >= 0 else -1.0
    return 0.5 * (a + b) + 0.5 * gam * s * (a - b)


@njit(cache=True)
def k_impeller(u, w, G, rc, imp_w, imp_id, jin, omega, sb, cb, chord, kcn, nblades,
               ftheta, fz, fn_blade, dz, h_band):
    """Blade-element force. fn_blade[k, i] = normal force per unit span per blade [N/m / rho]."""
    Nr, Nz = G.shape
    fn_blade[:, :] = 0.0
    for i in range(Nr):
        for j in range(Nz):
            if imp_w[i, j] <= 0:
                ftheta[i, j] = 0.0
                fz[i, j] = 0.0
                continue
            # leading edge meets the inflow swirl (cell just above the band, down-pumping);
            # swirl inside the band is mostly the blade's own wake
            v = G[i, jin[imp_id[i, j]]] / rc[i]
            wc = 0.5 * (w[i, j] + w[i, j + 1])
            vt = v - omega * rc[i]          # relative tangential velocity (blade frame)
            va = wc
            V2 = vt * vt + va * va
            V = np.sqrt(V2) + 1e-12
            sina = -(vt * sb - va * cb) / V  # plate normal n = (sin b, -cos b)
            # flat plate normal-force curve: attached 2*pi*sin(a), post-stall ~ K*sqrt(sin a)
            # (rotating blade -> delayed stall); K calibrated against literature Np
            sa = abs(sina)
            cn = min(2 * np.pi * sa, kcn * np.sqrt(sa))
            if sina < 0:
                cn = -cn
            fn = 0.5 * V2 * chord * cn   # per unit span, per unit density
            ftheta[i, j] = nblades * fn * sb * imp_w[i, j]
            fz[i, j] = -nblades * fn * cb * imp_w[i, j]
            # blade normal force per span: weight by axial fraction of the band
            fz_frac = imp_w[i, j] * 2 * np.pi * rc[i] * h_band  # = fr*fz overlap fraction
            fn_blade[imp_id[i, j], i] += fn * fz_frac * dz / h_band


@njit(cache=True)
def k_momentum(u, w, G, nut, rc, rf, dr, dz, vwin, nw_in, nw_out, nw_bot, nw_top,
               ftheta, fz, baf, trip, gam, dt, un, wn, Gn):
    Nr, Nz = G.shape
    # ---------------- u (radial) at faces k=1..Nr-1
    for k in range(1, Nr):
        r = rf[k]
        for j in range(Nz):
            # radial convective flux at cells k-1 and k
            iL = k - 1
            ut = 0.5 * (u[iL, j] + u[iL + 1, j])
            FL = rc[iL] * ut * blend(u[iL, j], u[iL + 1, j], ut, gam)
            iR = k
            ut = 0.5 * (u[iR, j] + u[iR + 1, j])
            FR = rc[iR] * ut * blend(u[iR, j], u[iR + 1, j], ut, gam)
            conv = (FR - FL) / (r * dr)
            # axial convective flux at corners j-1/2 and j+1/2
            if j < Nz - 1:
                wt = 0.5 * (w[k - 1, j + 1] + w[k, j + 1])
                Ft = wt * blend(u[k, j], u[k, j + 1], wt, gam)
            else:
                Ft = 0.0
            if j > 0:
                wt = 0.5 * (w[k - 1, j] + w[k, j])
                Fb = wt * blend(u[k, j - 1], u[k, j], wt, gam)
            else:
                Fb = 0.0
            conv += (Ft - Fb) / dz
            v = 0.5 * (G[k - 1, j] / rc[k - 1] + G[k, j] / rc[k])
            # viscous
            nuf = 0.5 * (nut[k - 1, j] + nut[k, j])
            VR = rc[iR] * nut[iR, j] * (u[iR + 1, j] - u[iR, j]) / dr
            VL = rc[iL] * nut[iL, j] * (u[iL + 1, j] - u[iL, j]) / dr
            visc = (VR - VL) / (r * dr) - nuf * u[k, j] / (r * r)
            if j < Nz - 1:
                nuc = 0.25 * (nut[k - 1, j] + nut[k, j] + nut[k - 1, j + 1] + nut[k, j + 1])
                Vt = nuc * (u[k, j + 1] - u[k, j]) / dz
            else:
                Vt = 0.5 * (nw_top[k - 1] + nw_top[k]) * (0.0 - u[k, j]) / (0.5 * dz)
            if j > 0:
                nuc = 0.25 * (nut[k - 1, j] + nut[k, j] + nut[k - 1, j - 1] + nut[k, j - 1])
                Vb = nuc * (u[k, j] - u[k, j - 1]) / dz
            else:
                Vb = 0.5 * (nw_bot[k - 1] + nw_bot[k]) * (u[k, j] - 0.0) / (0.5 * dz)
            visc += (Vt - Vb) / dz
            # tripod drag
            tr = 0.5 * (trip[k - 1, j] + trip[k, j])
            wcl = 0.25 * (w[k - 1, j] + w[k - 1, j + 1] + w[k, j] + w[k, j + 1])
            sp_ = np.sqrt(u[k, j] ** 2 + v * v + wcl * wcl)
            un[k, j] = u[k, j] + dt * (-conv + v * v / r + visc - tr * sp_ * u[k, j])
    # ---------------- w (axial) at faces l=1..Nz-1
    for i in range(Nr):
        for l in range(1, Nz):
            # radial convective flux at corners rf[i], rf[i+1]
            if i < Nr - 1:
                ut = 0.5 * (u[i + 1, l - 1] + u[i + 1, l])
                FR = rf[i + 1] * ut * blend(w[i, l], w[i + 1, l], ut, gam)
            else:
                FR = 0.0
            if i > 0:
                ut = 0.5 * (u[i, l - 1] + u[i, l])
                FL = rf[i] * ut * blend(w[i - 1, l], w[i, l], ut, gam)
            else:
                FL = 0.0
            conv = (FR - FL) / (rc[i] * dr)
            wt = 0.5 * (w[i, l] + w[i, l + 1])
            Ft = wt * blend(w[i, l], w[i, l + 1], wt, gam)
            wt = 0.5 * (w[i, l - 1] + w[i, l])
            Fb = wt * blend(w[i, l - 1], w[i, l], wt, gam)
            conv += (Ft - Fb) / dz
            # viscous radial
            if i < Nr - 1:
                nuc = 0.25 * (nut[i, l - 1] + nut[i, l] + nut[i + 1, l - 1] + nut[i + 1, l])
                VR = rf[i + 1] * nuc * (w[i + 1, l] - w[i, l]) / dr
            else:
                VR = rf[i + 1] * 0.5 * (nw_out[l - 1] + nw_out[l]) * (0.0 - w[i, l]) / (0.5 * dr)
            if i > 0:
                nuc = 0.25 * (nut[i, l - 1] + nut[i, l] + nut[i - 1, l - 1] + nut[i - 1, l])
                VL = rf[i] * nuc * (w[i, l] - w[i - 1, l]) / dr
            else:
                VL = rf[i] * 0.5 * (nw_in[l - 1] + nw_in[l]) * (w[i, l] - 0.0) / (0.5 * dr)
            visc = (VR - VL) / (rc[i] * dr)
            visc += (nut[i, l] * (w[i, l + 1] - w[i, l]) - nut[i, l - 1] * (w[i, l] - w[i, l - 1])) / (dz * dz)
            f = 0.5 * (fz[i, l - 1] + fz[i, l])
            tr = 0.5 * (trip[i, l - 1] + trip[i, l])
            v = G[i, l] / rc[i]
            ucl = 0.25 * (u[i, l] + u[i + 1, l] + u[i, l - 1] + u[i + 1, l - 1])
            sp_ = np.sqrt(w[i, l] ** 2 + v * v + ucl * ucl)
            wn[i, l] = w[i, l] + dt * (-conv + visc + f - tr * sp_ * w[i, l])
    # ---------------- G = r v (cells)
    for i in range(Nr):
        for j in range(Nz):
            # radial convection
            if i < Nr - 1:
                FR = rf[i + 1] * u[i + 1, j] * blend(G[i, j], G[i + 1, j], u[i + 1, j], gam)
                nuf = 0.5 * (nut[i, j] + nut[i + 1, j])
                TR = rf[i + 1] ** 3 * nuf * (G[i + 1, j] / rc[i + 1] ** 2 - G[i, j] / rc[i] ** 2) / dr
            else:
                FR = 0.0
                TR = rf[i + 1] ** 2 * nw_out[j] * (0.0 - G[i, j] / rc[i]) / (0.5 * dr)
            if i > 0:
                FL = rf[i] * u[i, j] * blend(G[i - 1, j], G[i, j], u[i, j], gam)
                nuf = 0.5 * (nut[i, j] + nut[i - 1, j])
                TL = rf[i] ** 3 * nuf * (G[i, j] / rc[i] ** 2 - G[i - 1, j] / rc[i - 1] ** 2) / dr
            else:
                FL = 0.0
                TL = rf[i] ** 2 * nw_in[j] * (G[i, j] / rc[i] - vwin[j]) / (0.5 * dr)
            conv = (FR - FL) / (rc[i] * dr)
            visc = (TR - TL) / (rc[i] * dr)
            if j < Nz - 1:
                Ft = w[i, j + 1] * blend(G[i, j], G[i, j + 1], w[i, j + 1], gam)
                Vt = 0.5 * (nut[i, j] + nut[i, j + 1]) * (G[i, j + 1] - G[i, j]) / dz
            else:
                Ft = 0.0
                Vt = nw_top[i] * (0.0 - G[i, j]) / (0.5 * dz)
            if j > 0:
                Fb = w[i, j] * blend(G[i, j - 1], G[i, j], w[i, j], gam)
                Vb = 0.5 * (nut[i, j] + nut[i, j - 1]) * (G[i, j] - G[i, j - 1]) / dz
            else:
                Fb = 0.0
                Vb = nw_bot[i] * (G[i, j] - 0.0) / (0.5 * dz)
            conv += (Ft - Fb) / dz
            visc += (Vt - Vb) / dz
            v = G[i, j] / rc[i]
            uc = 0.5 * (u[i, j] + u[i + 1, j])
            wc = 0.5 * (w[i, j] + w[i, j + 1])
            sp_ = np.sqrt(uc * uc + v * v + wc * wc)
            src = rc[i] * (ftheta[i, j] - baf[i, j] * v * abs(v) - trip[i, j] * sp_ * v)
            Gn[i, j] = G[i, j] + dt * (-conv + visc + src)


@njit(cache=True)
def vanleer(cU, cC, cD):
    # limited face value; cU upstream-upstream, cC upstream, cD downstream
    d1 = cC - cU
    d2 = cD - cC
    if d1 * d2 <= 0:
        return cC
    r = d1 / d2
    phi = (r + abs(r)) / (1 + abs(r))
    return cC + 0.5 * phi * d2


@njit(cache=True)
def k_scalar(u, w, c, nut, rc, rf, dr, dz, sct, dt, cn):
    Nr, Nz = c.shape
    for i in range(Nr):
        for j in range(Nz):
            # radial faces
            FR = 0.0
            DR = 0.0
            if i < Nr - 1:
                uf = u[i + 1, j]
                if uf >= 0:
                    cf = vanleer(c[i - 1, j] if i > 0 else c[i, j], c[i, j], c[i + 1, j])
                else:
                    cf = vanleer(c[i + 2, j] if i + 2 < Nr else c[i + 1, j], c[i + 1, j], c[i, j])
                FR = rf[i + 1] * uf * cf
                DR = rf[i + 1] * 0.5 * (nut[i, j] + nut[i + 1, j]) / sct * (c[i + 1, j] - c[i, j]) / dr
            FL = 0.0
            DL = 0.0
            if i > 0:
                uf = u[i, j]
                if uf >= 0:
                    cf = vanleer(c[i - 2, j] if i > 1 else c[i - 1, j], c[i - 1, j], c[i, j])
                else:
                    cf = vanleer(c[i + 1, j] if i + 1 < Nr else c[i, j], c[i, j], c[i - 1, j])
                FL = rf[i] * uf * cf
                DL = rf[i] * 0.5 * (nut[i, j] + nut[i - 1, j]) / sct * (c[i, j] - c[i - 1, j]) / dr
            Ft = 0.0
            Dt = 0.0
            if j < Nz - 1:
                wf = w[i, j + 1]
                if wf >= 0:
                    cf = vanleer(c[i, j - 1] if j > 0 else c[i, j], c[i, j], c[i, j + 1])
                else:
                    cf = vanleer(c[i, j + 2] if j + 2 < Nz else c[i, j + 1], c[i, j + 1], c[i, j])
                Ft = wf * cf
                Dt = 0.5 * (nut[i, j] + nut[i, j + 1]) / sct * (c[i, j + 1] - c[i, j]) / dz
            Fb = 0.0
            Db = 0.0
            if j > 0:
                wf = w[i, j]
                if wf >= 0:
                    cf = vanleer(c[i, j - 2] if j > 1 else c[i, j - 1], c[i, j - 1], c[i, j])
                else:
                    cf = vanleer(c[i, j + 1] if j + 1 < Nz else c[i, j], c[i, j], c[i, j - 1])
                Fb = wf * cf
                Db = 0.5 * (nut[i, j] + nut[i, j - 1]) / sct * (c[i, j] - c[i, j - 1]) / dz
            cn[i, j] = c[i, j] + dt * (-(FR - FL) / (rc[i] * dr) - (Ft - Fb) / dz
                                       + (DR - DL) / (rc[i] * dr) + (Dt - Db) / dz)


@njit(cache=True)
def k_div(u, w, rc, rf, dr, dz, div):
    Nr, Nz = div.shape
    for i in range(Nr):
        for j in range(Nz):
            div[i, j] = (rf[i + 1] * u[i + 1, j] - rf[i] * u[i, j]) / (rc[i] * dr) + (w[i, j + 1] - w[i, j]) / dz


@njit(cache=True)
def k_project(u, w, phi, dr, dz, dt):
    Nr, Nz = phi.shape
    for k in range(1, Nr):
        for j in range(Nz):
            u[k, j] -= dt * (phi[k, j] - phi[k - 1, j]) / dr
    for i in range(Nr):
        for l in range(1, Nz):
            w[i, l] -= dt * (phi[i, l] - phi[i, l - 1]) / dz


# ----------------------------------------------------------------------------- driver
def run_case(name, baffles, t_startup=C.T_STARTUP, t_mix=C.T_MIX, kcn=C.K_CN,
             out=None, verbose=True, gam=0.5):
    g = Grid()
    Nr, Nz = g.Nr, g.Nz
    imp_w, imp_id, h_band, baf, trip = build_sources(g, baffles)
    lu = build_poisson(g)
    u = np.zeros((Nr + 1, Nz)); w = np.zeros((Nr, Nz + 1)); G = np.zeros((Nr, Nz))
    un, wn, Gn = u.copy(), w.copy(), G.copy()
    c = np.zeros((Nr, Nz)); cn = c.copy()
    nut = np.full((Nr, Nz), C.NU)
    nw_in = np.zeros(Nz); nw_out = np.zeros(Nz); nw_bot = np.zeros(Nr); nw_top = np.zeros(Nr)
    ftheta = np.zeros((Nr, Nz)); fz = np.zeros((Nr, Nz))
    nimp = len(C.Z_IMPELLERS)
    fnb = np.zeros((nimp, Nr))
    div = np.zeros((Nr, Nz))
    sb, cb = np.sin(np.radians(C.PITCH_DEG)), np.cos(np.radians(C.PITCH_DEG))
    rc, rf, dr, dz = g.rc, g.rf, g.dr, g.dz
    vol = g.vol
    Vtot = vol.sum()

    # tracer blob
    tb = C.TRACER_BOX
    blob = ((g.RC > tb["r"][0]) & (g.RC < tb["r"][1]) & (g.ZC > tb["z"][0]) & (g.ZC < tb["z"][1]))
    c_inf = (vol * blob).sum() / Vtot
    probe_ij = {k: (int(np.argmin(abs(rc - p[0]))), int(np.argmin(abs(g.zc - p[1])))) for k, p in C.PROBES.items()}

    # z-index just below each impeller band (for flow number)
    jq = [int(np.searchsorted(g.zf, zi - h_band / 2 - 0.5 * dz)) for zi in C.Z_IMPELLERS]
    iq = int(np.searchsorted(rf, C.R_IMP))
    # first cell row above each impeller band (inflow for down-pumping blades)
    jin = np.array([int(np.searchsorted(g.zf, zi + h_band / 2)) for zi in C.Z_IMPELLERS], dtype=np.int64)

    t, step = 0.0, 0
    t_end = t_startup + t_mix
    injected = False
    snaps = {k: [] for k in ("t", "u", "v", "w", "nut", "c")}
    hist = {k: [] for k in ("t", "torque", "power", "Np", "Nq", "ekin", "swirl", "cov", "probes")}
    next_snap = 0.0
    dt = 1e-3
    t0 = time.time()
    while t < t_end - 1e-9:
        if not injected and t >= t_startup:
            c[blob] = 1.0
            injected = True
        ramp = min(1.0, t / T_RAMP)          # soft start (frequency converter)
        omega = C.OMEGA * ramp
        vwin = g.vwall_in * ramp
        k_turb(u, w, G, rc, dr, dz, g.r0, vwin, g.dwall, C.L_MIX_MAX, C.NU, nut,
               nw_in, nw_out, nw_bot, nw_top)
        if step % 5 == 0:
            vmax = max(np.abs(u).max(), np.abs(w).max(), np.abs(G / g.RC).max(), 0.5)
            nmax = max(nut.max(), nw_in.max(), nw_out.max(), nw_bot.max(), nw_top.max())
            dt = min(C.CFL * dr / vmax, 0.2 * min(dr, dz) ** 2 / nmax, 0.01)
            gam_eff = max(gam, 1.2 * vmax * dt / dr)
        k_impeller(u, w, G, rc, imp_w, imp_id, jin, omega, sb, cb, C.BLADE_W, kcn, C.N_BLADES,
                   ftheta, fz, fnb, dz, h_band)
        k_momentum(u, w, G, nut, rc, rf, dr, dz, vwin, nw_in, nw_out, nw_bot, nw_top,
                   ftheta, fz, baf, trip, gam_eff, dt, un, wn, Gn)
        if injected:
            # TVD (van Leer) explicit update needs a lower Courant number than momentum
            for _ in range(3):
                k_scalar(u, w, c, nut, rc, rf, dr, dz, C.SC_T, dt / 3, cn)
                c, cn = cn, c
        u, un = un, u
        w, wn = wn, w
        G, Gn = Gn, G
        k_div(u, w, rc, rf, dr, dz, div)
        rhs = (div / dt).ravel()
        rhs[0] = 0.0
        phi = lu.solve(rhs).reshape(Nr, Nz)
        k_project(u, w, phi, dr, dz, dt)
        t += dt
        step += 1

        if t >= next_snap - 1e-9:
            next_snap += C.SNAP_DT
            uc = 0.5 * (u[:-1] + u[1:]); wc = 0.5 * (w[:, :-1] + w[:, 1:]); v = G / g.RC
            torque = C.RHO * (ftheta * g.RC * vol).sum()
            power = torque * omega
            Np = power / (C.RHO * max(ramp * C.N_REV, 1e-3)**3 * C.D_IMP**5) / nimp
            Q = [(-np.minimum(w[:iq, jj], 0) * 2 * np.pi * rc[:iq] * dr).sum() for jj in jq]
            Nq = np.mean(Q) / (C.N_REV * C.D_IMP**3)
            ek = 0.5 * C.RHO * ((uc**2 + v**2 + wc**2) * vol).sum()
            swirl = (v * vol).sum() / Vtot / (C.OMEGA * C.R_IMP)
            if injected:
                cm = (c * vol).sum() / Vtot
                cov = np.sqrt(((c - cm) ** 2 * vol).sum() / Vtot) / max(cm, 1e-12)
                pr = [c[ij] / c_inf for ij in probe_ij.values()]
            else:
                cov, pr = np.nan, [np.nan] * len(probe_ij)
            for k_, val in zip(hist.keys(), (t, torque, power, Np, Nq, ek, swirl, cov, pr)):
                hist[k_].append(val)
            for k_, val in zip(("t", "u", "v", "w", "nut", "c"), (t, uc, v, wc, nut, c / c_inf)):
                snaps[k_].append(np.asarray(val, dtype=np.float32) if k_ != "t" else val)
            if verbose and int(t / C.SNAP_DT) % 20 == 0:
                print(f"[{name}] t={t:7.2f}s dt={dt*1e3:5.2f}ms Np={Np:5.2f} Nq={Nq:5.2f} "
                      f"swirl={swirl:5.3f} CoV={cov:7.3f}  wall={time.time()-t0:6.0f}s", flush=True)

    res = dict(
        name=name, baffles=baffles, rc=rc, zc=g.zc, rf=rf, zf=g.zf, r0=g.r0, R=g.R, H=g.H,
        t=np.array(snaps["t"]), u=np.array(snaps["u"]), v=np.array(snaps["v"]),
        w=np.array(snaps["w"]), nut=np.array(snaps["nut"]), c=np.array(snaps["c"]),
        h_t=np.array(hist["t"]), torque=np.array(hist["torque"]), power=np.array(hist["power"]),
        Np=np.array(hist["Np"]), Nq=np.array(hist["Nq"]), ekin=np.array(hist["ekin"]),
        swirl=np.array(hist["swirl"]), cov=np.array(hist["cov"]), probes=np.array(hist["probes"]),
        probe_names=np.array(list(C.PROBES.keys())), t_inj=t_startup, c_inf=c_inf,
        fn_blade=fnb * C.RHO,   # N/m per blade, per impeller (at final time)
        kcn=kcn, h_band=h_band,
    )
    if out:
        np.savez_compressed(out, **res)
    return res


if __name__ == "__main__":
    import os
    name = sys.argv[1]
    os.makedirs("results", exist_ok=True)
    run_case(name, C.CASES[name]["baffles"], out=f"results/cfd_{name}.npz")
