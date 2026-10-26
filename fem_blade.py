"""
FEM-analyse schoep (vlakke plaat onder 45 graden) met belasting uit de CFD.

- 3D solid model, kwadratische tetraëders (TetP2, 10 knopen), scikit-fem
- Inklemming aan de schoepwortel (las op naaf)
- Belasting: normaaldruk per spanwijdte-positie q(r) uit het blade-element model
  (CFD, eindtoestand), koordegewijs lineair verdeeld (drukpunt op 1/3 koorde),
  plus centrifugaalkracht.
- Lastgevallen: ontwerp (gemiddeld x DYN_FACTOR) en vermoeiing (fluctuatie +/-)
- Modale analyse: droog en nat (toegevoegde watermassa, strip-theorie)
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import (MeshTet, ElementTetP2, ElementVector, Basis, FacetBasis, BilinearForm,
                   LinearForm, asm, condense, solve)
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import sym_grad, dot

import config as C

L = C.R_IMP - C.R_HUB      # span
W = C.BLADE_W
T = C.BLADE_T


def make_model(nx=32, ny=12, nz=2):
    m = MeshTet.init_tensor(np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1),
                            np.linspace(-T / 2, T / 2, nz + 1))
    e = ElementVector(ElementTetP2())
    ib = Basis(m, e, intorder=4)
    lam, mu = lame_parameters(C.E_STEEL, C.NU_STEEL)
    K = asm(linear_elasticity(lam, mu), ib)

    @BilinearForm
    def mass(u, v, w):
        return C.RHO_STEEL * dot(u, v)

    M = asm(mass, ib)
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    return m, ib, K, M, D, lam, mu


def load_vector(m, ib, q_of_x, centrifugal=True):
    top = m.facets_satisfying(lambda x: np.isclose(x[2], T / 2))
    fb = FacetBasis(m, ib.elem, facets=top, intorder=4)

    @LinearForm
    def trac(v, w):
        x = w.x
        p = np.interp(x[0], q_of_x[0], q_of_x[1]) / W * 2.0 * (1.0 - x[1] / W)
        return -p * v.value[2]

    f = asm(trac, fb)
    if centrifugal:
        @LinearForm
        def cf(v, w):
            return C.RHO_STEEL * C.OMEGA**2 * (C.R_HUB + w.x[0]) * v.value[0]
        f = f + asm(cf, ib)
    return f


def von_mises(ib, u, lam, mu):
    uh = ib.interpolate(u)
    eps = sym_grad(uh)
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = 2 * mu * eps
    for i in range(3):
        s[i, i] = s[i, i] + lam * tr
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1])**2 + (s[1, 1] - s[2, 2])**2 + (s[2, 2] - s[0, 0])**2)
                 + 3 * (s[0, 1]**2 + s[1, 2]**2 + s[0, 2]**2))
    sb = ib.with_element(ElementTetP2())
    vm_nodal = sb.project(vm)
    sxx_nodal = sb.project(s[0, 0])
    return sb, vm_nodal, sxx_nodal, vm


def surface_grid(sb, field, z=T / 2):
    """Structured (x, y) grid of a nodal P2 field on plane z (mesh vertices only)."""
    X = sb.mesh.p
    field = field[sb.nodal_dofs[0]]
    xs, ys = np.unique(X[0].round(9)), np.unique(X[1].round(9))
    sel = np.isclose(X[2], z)
    ix = np.searchsorted(xs, X[0, sel].round(9))
    iy = np.searchsorted(ys, X[1, sel].round(9))
    F = np.zeros((len(xs), len(ys)))
    F[ix, iy] = field[sel]
    return xs, ys, F


def run(cfd_file="results/cfd_geschot.npz", out="results/fem_blade.npz"):
    d = np.load(cfd_file)
    rc = d["rc"]
    fnb = d["fn_blade"]                     # (n_imp, Nr) N/m per blade
    k = int(np.argmax(fnb.sum(axis=1)))     # zwaarst belaste waaier
    q = fnb[k]
    sel = (rc >= C.R_HUB) & (rc <= C.R_IMP)
    xq = np.concatenate([[0.0], rc[sel] - C.R_HUB, [L]])
    qq = np.concatenate([[q[sel][0]], q[sel], [q[sel][-1]]])
    F_tot = np.trapezoid(qq, xq)
    M_root = np.trapezoid(qq * xq, xq)

    m, ib, K, M, D, lam, mu = make_model()
    print(f"FEM: {m.nelements} TetP2 elementen, {ib.N} vrijheidsgraden")

    # --- design load: mean * dynamic factor
    f_mean = load_vector(m, ib, (xq, qq))
    u_mean = solve(*condense(K, f_mean, D=D))
    sb, vm_mean, sxx_mean, vm_qp = von_mises(ib, u_mean, lam, mu)
    u_des = C.DYN_FACTOR * u_mean        # linear (centrifugal is small; conservative)
    vm_des = C.DYN_FACTOR * vm_mean

    # nominal root bending stress (beam theory) for comparison
    Wb = W * T**2 / 6
    sig_nom = M_root / Wb

    # FEM structural stress at the root, mid-chord (away from the corner singularities)
    xs, ys, SXX = surface_grid(sb, sxx_mean, z=T / 2)
    mid = (ys > 0.2 * W) & (ys < 0.8 * W)
    s_root_fem = np.abs(SXX[0, mid]).max()
    # fatigue (EN 1993-1-9): nominal stress range at the root weld from load fluctuation,
    # constant-amplitude fatigue limit dsD = 0.737*FAT with partial factor 1.35
    d_sig_fat = 2 * C.FLUCT_AMP * sig_nom
    ds_L = 0.737 * C.FAT_CLASS / 1.35
    t_req = T * np.sqrt(max(d_sig_fat / ds_L, C.DYN_FACTOR * sig_nom / (C.SY_316L / 1.5), 0))

    # --- modal analysis
    I = np.setdiff1d(np.arange(ib.N), D)
    Kc, Mc = K[I][:, I], M[I][:, I]
    lam_e, vec = spla.eigsh(Kc, k=6, M=Mc, sigma=0, which="LM")
    f_dry = np.sqrt(lam_e) / (2 * np.pi)
    modes = np.zeros((6, ib.N))
    modes[:, I] = vec.T
    # wet: added mass of water on a strip of width W moving normal to plate
    m_plate = C.RHO_STEEL * T
    m_add = C.RHO * np.pi * W / 4
    # out-of-plane modes (dominant z-motion) get the added mass, in-plane modes hardly
    zfrac = np.array([(ib.split(md)[2][0] ** 2).sum() / (md ** 2).sum() for md in modes])
    f_wet = np.where(zfrac > 0.5, f_dry * np.sqrt(m_plate / (m_plate + m_add)), f_dry)
    f_bpf = C.N_BLADES * C.N_REV
    f_baffle = C.N_BAFFLES * C.N_REV

    res = dict(
        imp_index=k, xq=xq, qq=qq, F_tot=F_tot, M_root=M_root, sig_nom=sig_nom,
        vm_max_mean=vm_qp.max(), vm_max_des=C.DYN_FACTOR * vm_qp.max(), s_root_fem=s_root_fem, d_sig_fat=d_sig_fat, t_req=t_req,
        ds_L=ds_L, f_dry=f_dry, f_wet=f_wet, f_bpf=f_bpf, f_baffle=f_baffle, f_rot=C.N_REV,
        tip_defl=np.abs(ib.split(u_mean)[2][0]).max(), zfrac=zfrac,
    )
    # surface grids (mesh vertices) for visualisation
    res["xs"], res["ys"], _ = surface_grid(sb, vm_mean)
    for c_i, cn in enumerate("xyz"):
        res["u" + cn] = surface_grid(sb, ib.split(u_mean)[c_i][0])[2]
    res["vm_grid"] = np.maximum(surface_grid(sb, vm_mean, T / 2)[2], surface_grid(sb, vm_mean, -T / 2)[2])
    res["mode_uz"] = np.array([surface_grid(sb, ib.split(md)[2][0])[2] for md in modes])
    res["mode_uy"] = np.array([surface_grid(sb, ib.split(md)[1][0])[2] for md in modes])
    np.savez_compressed(out, **res)
    print(f"Waaier {k+1}: totale normaalkracht/schoep {F_tot/1e3:.2f} kN, wortelmoment {M_root/1e3:.2f} kNm")
    print(f"Nominale buigspanning wortel (balk) {sig_nom/1e6:.1f} MPa")
    print(f"FEM von Mises max (gemiddeld) {vm_qp.max()/1e6:.1f} MPa, ontwerp x{C.DYN_FACTOR}: {C.DYN_FACTOR*vm_qp.max()/1e6:.1f} MPa "
          f"(vloeigrens {C.SY_316L/1e6:.0f} MPa -> SF {C.SY_316L/(C.DYN_FACTOR*vm_qp.max()):.2f})")
    print(f"FEM wortelspanning midden koorde {s_root_fem/1e6:.1f} MPa")
    print(f"Vermoeiing: spanningswisseling {d_sig_fat/1e6:.1f} MPa vs grens {ds_L/1e6:.1f} MPa -> "
          f"benodigde dikte {t_req*1e3:.1f} mm")
    print(f"Tipuitbuiging {res['tip_defl']*1e3:.2f} mm")
    print("Eigenfrequenties droog [Hz]:", f_dry.round(1))
    print("Eigenfrequenties nat (schatting) [Hz]:", f_wet.round(1), " BPF:", f_bpf, "Hz")
    return res


if __name__ == "__main__":
    run()
