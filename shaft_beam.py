"""
As (koker 7 m) als Euler-Bernoulli balk-FEM.

Randvoorwaarden: boven ingeklemd (aandrijving/tandwielkast met twee lagers),
onder scharnierend in het driepoot-steunlager (alleen radiaal).
Belasting: hydraulische radiale kracht per waaier (fluctuerend, empirisch)
Uitkomsten: doorbuiging, buigspanning, lagerreactie -> pootkrachten driepoot,
eerste eigenfrequentie (kritisch toerental) droog/nat.
"""
import numpy as np
import scipy.linalg as sla

import config as C

E = C.E_STEEL
DO, DI = C.SHAFT_OD, C.SHAFT_OD - 2 * C.SHAFT_WALL
A = np.pi * (DO**2 - DI**2) / 4
I = np.pi * (DO**4 - DI**4) / 64
WB = I / (DO / 2)


def beam_mats(z, m_line):
    n = len(z)
    K = np.zeros((2 * n, 2 * n)); M = np.zeros_like(K)
    for e in range(n - 1):
        h = z[e + 1] - z[e]
        ke = E * I / h**3 * np.array([[12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h],
                                      [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
        me = m_line * h / 420 * np.array([[156, 22*h, 54, -13*h], [22*h, 4*h*h, 13*h, -3*h*h],
                                          [54, 13*h, 156, -22*h], [-13*h, -3*h*h, -22*h, 4*h*h]])
        idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    return K, M


def run(cfd_file="results/cfd_geschot.npz", out="results/shaft.npz"):
    d = np.load(cfd_file)
    torque_imp = np.mean(d["torque"][-40:]) / len(C.Z_IMPELLERS)
    # empirical hydraulic side load (unbalanced flow): F_r ~ 0.25 * T / R_imp
    F_r = 0.25 * torque_imp / C.R_IMP

    z = np.linspace(C.Z_BEARING, C.H_LIQ, 141)     # z measured from tank floor
    m_steel = C.RHO_STEEL * A
    m_add = C.RHO * np.pi * DO**2 / 4              # added water mass around the koker
    blade_m = C.RHO_STEEL * (C.R_IMP - C.R_HUB) * C.BLADE_W * C.BLADE_T
    m_imp = C.N_BLADES * blade_m + 60.0            # + naaf
    m_imp_wet = m_imp + C.N_BLADES * C.RHO * np.pi * C.BLADE_W**2 / 4 * (C.R_IMP - C.R_HUB) * 0.5

    res = {}
    fixed = [2 * (len(z) - 1), 2 * (len(z) - 1) + 1, 0]  # top: w, theta ; bottom: w
    free = np.setdiff1d(np.arange(2 * len(z)), fixed)
    nodes_imp = [int(np.argmin(abs(z - zi))) for zi in C.Z_IMPELLERS]
    for tag, ml, mi in (("droog", m_steel, m_imp), ("nat", m_steel + m_add, m_imp_wet)):
        K, M = beam_mats(z, ml)
        for n in nodes_imp:
            M[2 * n, 2 * n] += mi
        lam = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True,
                       subset_by_index=[0, 2])
        res["f_" + tag] = np.sqrt(lam) / (2 * np.pi)

    # static: both impeller side loads in same direction (worst case) x dynamic factor
    K, _ = beam_mats(z, m_steel)
    f = np.zeros(2 * len(z))
    for n in nodes_imp:
        f[2 * n] += C.DYN_FACTOR * F_r
    u = np.zeros_like(f)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    R = K @ u - f
    R_bottom = R[0]
    defl = u[0::2]
    # bending moment from statics (bottom reaction + impeller loads below the section)
    Mb = R_bottom * (z - z[0])
    for n in nodes_imp:
        Mb = Mb + f[2 * n] * np.clip(z - z[n], 0, None)
    Mb = -Mb
    sig_b = np.abs(Mb).max() / WB
    tau_t = (np.mean(d["torque"][-40:]) * C.DYN_FACTOR) * (DO / 2) / (2 * I)
    sig_vm = np.sqrt(sig_b**2 + 3 * tau_t**2)

    # tripod: 3 legs at 120 deg, inclination phi; horizontal load H along leg 1 direction
    phi = np.arctan2(C.Z_BEARING, C.TRIPOD_R - C.SHAFT_OD / 2)
    ang = np.radians([0, 120, 240])
    Amat = np.vstack([np.cos(phi) * np.cos(ang), np.cos(phi) * np.sin(ang), np.sin(phi) * np.ones(3)])
    leg_forces = []
    for th in np.radians(np.arange(0, 360, 10)):
        H = np.array([abs(R_bottom) * np.cos(th), abs(R_bottom) * np.sin(th), 0.0])
        leg_forces.append(np.linalg.solve(Amat, H))
    leg_forces = np.array(leg_forces)

    res.update(dict(z=z, defl=defl, Mb=Mb, F_r=F_r, torque_imp=torque_imp, R_bottom=R_bottom,
                    sig_b=sig_b, tau_t=tau_t, sig_vm=sig_vm, leg_max=np.abs(leg_forces).max(),
                    leg_phi_deg=np.degrees(phi), f_rot=C.N_REV, m_imp=m_imp))
    np.savez_compressed(out, **res)
    print(f"Koppel per waaier {torque_imp/1e3:.2f} kNm; radiale hydraulische kracht {F_r/1e3:.2f} kN/waaier")
    print(f"Max doorbuiging {np.abs(defl).max()*1e3:.2f} mm; buigspanning {sig_b/1e6:.1f} MPa; "
          f"torsie {tau_t/1e6:.1f} MPa; VM {sig_vm/1e6:.1f} MPa")
    print(f"Reactie driepootlager {abs(R_bottom)/1e3:.2f} kN; max pootkracht {res['leg_max']/1e3:.2f} kN "
          f"(poothoek {res['leg_phi_deg']:.0f} gr)")
    print(f"Eigenfrequenties as droog {res['f_droog'].round(2)} Hz, nat {res['f_nat'].round(2)} Hz; "
          f"toerental {C.N_REV:.2f} Hz -> ratio {res['f_nat'][0]/C.N_REV:.1f}")
    return res


if __name__ == "__main__":
    run()
