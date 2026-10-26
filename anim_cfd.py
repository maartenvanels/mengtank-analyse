"""
Animaties CFD:
  1. stroming_opstart.mp4  - opbouw stromingspatroon (meridionaal + swirl), beide cases
  2. menging_tracer.mp4    - tracer-injectie en homogenisatie + menggraad (CoV) in de tijd
  3. deeltjes_3d.mp4       - 3D deeltjesbanen in het (quasi)stationaire stromingsveld
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Polygon
from scipy.interpolate import RegularGridInterpolator

import config as C

OUT = "animaties"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white"})
CASES = ["geschot", "ongeschot"]
LABEL = {k: C.CASES[k]["label"] for k in CASES}


def load():
    return {k: dict(np.load(f"results/cfd_{k}.npz")) for k in CASES}


def mirror(rc, F):
    """(Nr,Nz) field on r>0 -> full section x in [-R, R]."""
    x = np.concatenate([-rc[::-1], rc])
    return x, np.concatenate([F[::-1], F], axis=0)


def draw_hardware(ax, baffles, color="k"):
    R, H, r0 = C.T_DIAM / 2, C.H_LIQ, C.SHAFT_OD / 2
    ax.plot([-R, -R, R, R], [H, 0, 0, H], color=color, lw=1.5)
    ax.plot([-R, R], [H, H], color=color, lw=1.5)
    ax.add_patch(Rectangle((-r0, C.Z_BEARING), 2 * r0, H - C.Z_BEARING, fc="0.55", ec=color, lw=0.6, zorder=5))
    # tripod legs
    for s in (-1, 1):
        ax.plot([s * r0, s * C.TRIPOD_R], [C.Z_BEARING, 0], color="0.3", lw=2, zorder=5)
    hb = C.BLADE_W * np.sin(np.radians(C.PITCH_DEG))
    for zi in C.Z_IMPELLERS:
        for s in (-1, 1):
            # 45 deg blade seen edge-on (projection)
            ax.add_patch(Polygon([[s * C.R_HUB, zi - hb / 2], [s * C.R_IMP, zi - hb / 2],
                                  [s * C.R_IMP, zi + hb / 2], [s * C.R_HUB, zi + hb / 2]],
                                 fc="0.25", ec=color, lw=0.5, zorder=6))
    if baffles:
        for s in (-1, 1):
            ax.add_patch(Rectangle((s * R - (C.BAFFLE_W if s > 0 else 0), 0.15), C.BAFFLE_W, H - 0.3,
                                   fc="none", ec="0.2", ls="--", lw=0.8, zorder=6))
    ax.set_aspect("equal")
    ax.set_xlim(-R - 0.1, R + 0.1)
    ax.set_ylim(-0.1, H + 0.1)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks(range(0, 8))
    ax.set_xlabel("r [m]")


# ----------------------------------------------------------------------------- 1. opstart
def anim_startup(D, t_max=90.0, fps=12):
    a = D["geschot"]
    t = a["t"]
    frames = np.where(t <= t_max)[0][::1]
    fig, axs = plt.subplots(1, 4, figsize=(15, 7.2), constrained_layout=True)
    vmax_m, vmax_s = 3.5, 0.6
    ims, cbs = [], []

    def draw(fi):
        for ax in axs:
            ax.clear()
        for c_i, case in enumerate(CASES):
            d = D[case]
            rc, zc = d["rc"], d["zc"]
            u, w, v = d["u"][fi], d["w"][fi], d["v"][fi]
            spd = np.sqrt(u**2 + w**2)
            x, S = mirror(rc, spd)
            ax = axs[c_i]
            pc = ax.pcolormesh(x, zc, S.T, cmap="turbo", vmin=0, vmax=vmax_m, shading="auto")
            # streamlines of meridional flow (mirror u sign on left half)
            U = np.concatenate([-u[::-1], u], axis=0).T
            Wm = np.concatenate([w[::-1], w], axis=0).T
            xs = np.linspace(x[0], x[-1], 80)
            zs = np.linspace(zc[0], zc[-1], 200)
            Ui = RegularGridInterpolator((zc, x), U)(np.stack(np.meshgrid(zs, xs, indexing="ij"), -1))
            Wi = RegularGridInterpolator((zc, x), Wm)(np.stack(np.meshgrid(zs, xs, indexing="ij"), -1))
            ax.streamplot(xs, zs, Ui, Wi, color="w", density=1.3, linewidth=0.6, arrowsize=0.7)
            draw_hardware(ax, d["baffles"])
            ax.set_title(f"{LABEL[case]}\nmeridionale snelheid [m/s]")
            ax2 = axs[2 + c_i]
            xv, V = mirror(rc, v / (C.OMEGA * C.R_IMP))
            ax2.pcolormesh(xv, zc, V.T, cmap="magma", vmin=0, vmax=vmax_s, shading="auto")
            draw_hardware(ax2, d["baffles"], color="w")
            ax2.set_facecolor("k")
            ax2.set_title(f"{LABEL[case]}\nswirl  v$_\\theta$ / v$_{{tip}}$ [-]")
        fig.suptitle(f"Opbouw stromingspatroon na start roerder (100 omw/min)    t = {t[fi]:5.1f} s",
                     fontsize=12, fontweight="bold")
        return []

    draw(frames[-1])
    sm1 = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, vmax_m))
    sm2 = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax_s))
    fig.colorbar(sm1, ax=[axs[0], axs[1]], location="bottom", shrink=0.8, label="meridionale snelheid |u_rz| [m/s]")
    fig.colorbar(sm2, ax=[axs[2], axs[3]], location="bottom", shrink=0.8, label="swirl v_theta / v_tip [-]")
    fig.savefig(f"{OUT}/stroming_eindtoestand.png", dpi=110)
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(f"{OUT}/stroming_opstart.mp4", writer=FFMpegWriter(fps=fps, bitrate=2500), dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. menging
def anim_mixing(D, fps=12, every=1):
    a = D["geschot"]
    t = a["t"]
    ti = a["t_inj"]
    idx = np.where(t >= ti - 1e-6)[0][::every]
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], height_ratios=[1, 1])
    axc = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
    axv = fig.add_subplot(gs[0, 2])
    axp = fig.add_subplot(gs[1, 2])
    cmap = plt.cm.viridis
    tmix = {}
    for case in CASES:
        d = D[case]
        ht, cov = d["h_t"], d["cov"]
        ok = ~np.isnan(cov)
        below = np.where(ok & (cov > 0.05))[0]
        tmix[case] = (ht[below[-1]] - ti) if len(below) and below[-1] + 1 < len(ht) else np.nan

    def draw(fi):
        for ax in axc + [axv, axp]:
            ax.clear()
        tt = t[fi] - ti
        for k, case in enumerate(CASES):
            d = D[case]
            x, Cf = mirror(d["rc"], d["c"][fi])
            axc[k].pcolormesh(x, d["zc"], np.clip(Cf.T, 0, 2), cmap=cmap, vmin=0, vmax=2, shading="gouraud")
            draw_hardware(axc[k], d["baffles"], color="w")
            for nm, (pr, pz) in C.PROBES.items():
                axc[k].plot(pr, pz, "o", ms=4, mfc="none", mec="r")
            axc[k].set_title(f"{LABEL[case]}\nc / c$_\\infty$  (1 = homogeen)")
            ht = d["h_t"] - ti
            m = (ht >= 0) & (ht <= tt)
            axv.semilogy(ht[m], d["cov"][m], lw=2, label=LABEL[case] +
                         (f"  (t$_{{95}}$ = {tmix[case]:.0f} s)" if not np.isnan(tmix[case]) else "  (t$_{95}$ > sim.)"))
            ls = "-" if case == "geschot" else "--"
            for p_i, nm in enumerate(d["probe_names"]):
                axp.plot(ht[m], d["probes"][m, p_i], ls, lw=1.2, color=f"C{p_i}",
                         label=nm if case == "geschot" else None)
        axv.axhline(0.05, color="k", ls=":", lw=1)
        axv.text(2, 0.055, "95% homogeen (CoV = 0.05)", fontsize=8)
        axv.set_xlim(0, t[idx[-1]] - ti); axv.set_ylim(0.01, 30)
        axv.set_ylabel("CoV concentratie [-]"); axv.set_xlabel("tijd na injectie [s]")
        axv.set_title("Menggraad over het hele tankvolume"); axv.legend(loc="upper right", fontsize=8)
        axv.grid(True, which="both", alpha=0.3)
        axp.axhspan(0.95, 1.05, color="g", alpha=0.12)
        axp.set_xlim(0, t[idx[-1]] - ti); axp.set_ylim(0, 2.5)
        axp.set_xlabel("tijd na injectie [s]"); axp.set_ylabel("c / c$_\\infty$")
        axp.set_title("Monitorpunten (— geschot, -- ongeschot)")
        axp.legend(fontsize=7, ncol=2, loc="upper right"); axp.grid(alpha=0.3)
        fig.suptitle(f"Tracer-injectie onder het deksel en homogenisatie    t = {tt:5.1f} s na injectie",
                     fontsize=12, fontweight="bold")
        return []

    draw(idx[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2))
    fig.colorbar(sm, ax=axc, location="bottom", shrink=0.7, label="c / c_inf")
    fig.savefig(f"{OUT}/menging_eindtoestand.png", dpi=110)
    anim = FuncAnimation(fig, draw, frames=idx, blit=False)
    anim.save(f"{OUT}/menging_tracer.mp4", writer=FFMpegWriter(fps=fps, bitrate=2500), dpi=90)
    plt.close(fig)
    return tmix


# ----------------------------------------------------------------------------- 3. deeltjes 3D
class Field:
    def __init__(self, d, fi):
        rc, zc = d["rc"], d["zc"]
        self.r0, self.R, self.H = float(d["r0"]), float(d["R"]), float(d["H"])
        kw = dict(bounds_error=False, fill_value=None)
        self.fu = RegularGridInterpolator((rc, zc), d["u"][fi], **kw)
        self.fv = RegularGridInterpolator((rc, zc), d["v"][fi], **kw)
        self.fw = RegularGridInterpolator((rc, zc), d["w"][fi], **kw)
        self.fn = RegularGridInterpolator((rc, zc), d["nut"][fi], **kw)
        # random-walk drift (Ito): dr = (dD/dr + D/r) dt, dz = dD/dz dt, D = nu_t / Sc_t
        Dt = d["nut"][fi].astype(float) / C.SC_T
        gr, gz = np.gradient(Dt, rc, zc)
        self.fdr = RegularGridInterpolator((rc, zc), gr + Dt / rc[:, None], **kw)
        self.fdz = RegularGridInterpolator((rc, zc), gz, **kw)

    def vel(self, r, z):
        p = np.stack([r, z], -1)
        return self.fu(p) + self.fdr(p), self.fv(p), self.fw(p) + self.fdz(p), self.fn(p)


def advect(F, r, th, z, dt, rng):
    u, v, w, nt = F.vel(r, z)
    rm, zm = r + 0.5 * dt * u, z + 0.5 * dt * w
    rm = np.clip(rm, F.r0 + 1e-3, F.R - 1e-3); zm = np.clip(zm, 1e-3, F.H - 1e-3)
    u2, v2, w2, nt2 = F.vel(rm, zm)
    sig = np.sqrt(2 * np.clip(nt2, 0, None) / C.SC_T * dt)
    r = r + dt * u2 + sig * rng.standard_normal(r.size)
    z = z + dt * w2 + sig * rng.standard_normal(r.size)
    th = th + dt * v2 / np.maximum(rm, F.r0) + sig * rng.standard_normal(r.size) / np.maximum(rm, F.r0)
    # reflect at walls
    r = np.where(r < F.r0, 2 * F.r0 - r, r); r = np.where(r > F.R, 2 * F.R - r, r)
    z = np.where(z < 0, -z, z); z = np.where(z > F.H, 2 * F.H - z, z)
    return r, th, z


def anim_particles(D, n=2500, t_sim=90.0, dt=0.1, sub=3, fps=15):
    rng = np.random.default_rng(1)
    # uniform in volume
    R, H, r0 = C.T_DIAM / 2, C.H_LIQ, C.SHAFT_OD / 2
    r_init = np.sqrt(rng.uniform(r0**2, R**2, n))
    th_init = rng.uniform(0, 2 * np.pi, n)
    z_init = rng.uniform(0, H, n)
    col = np.where(z_init > H / 2, 0, 1)
    colors = np.array([[0.93, 0.45, 0.10, 0.85], [0.12, 0.40, 0.80, 0.85]])[col]
    st = {}
    for case in CASES:
        fi_steady = int(np.searchsorted(D[case]["t"], D[case]["t_inj"])) - 1
        st[case] = dict(F=Field(D[case], fi_steady),
                        r=r_init.copy(), th=th_init.copy(), z=z_init.copy())
    nfr = int(t_sim / (dt * sub))
    fig = plt.figure(figsize=(13, 7.4))
    axs = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    stats = {case: [] for case in CASES}

    def tank(ax, ang, baffles):
        tt = np.linspace(0, 2 * np.pi, 60)
        for zz in (0, H):
            ax.plot(R * np.cos(tt), R * np.sin(tt), zz, color="0.5", lw=0.6)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            ax.plot([R * np.cos(a)] * 2, [R * np.sin(a)] * 2, [0, H], color="0.8", lw=0.4)
        ax.plot([0, 0], [0, 0], [C.Z_BEARING, H], color="0.2", lw=4)
        for a in np.radians([0, 120, 240]):
            ax.plot([0, C.TRIPOD_R * np.cos(a)], [0, C.TRIPOD_R * np.sin(a)], [C.Z_BEARING, 0], color="0.2", lw=2.5)
        hb = C.BLADE_W * np.sin(np.radians(C.PITCH_DEG)) / 2
        for zi in C.Z_IMPELLERS:
            for b in range(C.N_BLADES):
                a = ang + 2 * np.pi * b / C.N_BLADES
                rr = np.array([C.R_HUB, C.R_IMP])
                da = 0.2 / rr
                X = np.array([[rr[0] * np.cos(a - da[0]), rr[1] * np.cos(a - da[1])],
                              [rr[0] * np.cos(a + da[0]), rr[1] * np.cos(a + da[1])]])
                Y = np.array([[rr[0] * np.sin(a - da[0]), rr[1] * np.sin(a - da[1])],
                              [rr[0] * np.sin(a + da[0]), rr[1] * np.sin(a + da[1])]])
                Z = np.array([[zi - hb, zi - hb], [zi + hb, zi + hb]])
                ax.plot_surface(X, Y, Z, color="0.3", alpha=0.9)
        if baffles:
            for a in np.radians([45, 135, 225, 315]):
                ax.plot_surface(np.array([[R - C.BAFFLE_W, R]] * 2) * np.cos(a),
                                np.array([[R - C.BAFFLE_W, R]] * 2) * np.sin(a),
                                np.array([[0.15, 0.15], [H - 0.15, H - 0.15]]), color="0.6", alpha=0.35)

    def draw(fi):
        for k, case in enumerate(CASES):
            s = st[case]
            if fi > 0:
                for _ in range(sub):
                    s["r"], s["th"], s["z"] = advect(s["F"], s["r"], s["th"], s["z"], dt, rng)
            ax = axs[k]
            ax.clear()
            tsim = fi * dt * sub
            tank(ax, C.OMEGA * tsim * 0.05, D[case]["baffles"])   # blades drawn slowed down
            x, y = s["r"] * np.cos(s["th"]), s["r"] * np.sin(s["th"])
            ax.scatter(x, y, s["z"], c=colors, s=3, depthshade=False, lw=0)
            # fraction of orange (top) particles in bottom half -> 0.5 = perfectly mixed
            frac = np.mean(s["z"][col == 0] < H / 2)
            stats[case].append(frac)
            ax.set_title(f"{LABEL[case]}\nbovenste helft (oranje) nu in onderste helft: {100*frac:4.1f}%  (ideaal 50%)",
                         fontsize=10)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, H)
            ax.set_box_aspect((2 * R, 2 * R, H))
            ax.view_init(elev=14, azim=-60 + 0.3 * fi)
            ax.set_axis_off()
        fig.suptitle(f"3D deeltjesbanen (gemiddeld veld + turbulente dispersie)   t = {fi*dt*sub:5.1f} s",
                     fontsize=12, fontweight="bold")
        return []

    anim = FuncAnimation(fig, draw, frames=nfr, blit=False)
    anim.save(f"{OUT}/deeltjes_3d.mp4", writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=90)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    import os, sys, json
    os.makedirs(OUT, exist_ok=True)
    D = load()
    which = sys.argv[1:] or ["startup", "mixing", "particles"]
    if "startup" in which:
        anim_startup(D)
    if "mixing" in which:
        tm = anim_mixing(D)
        json.dump({k: float(v) for k, v in tm.items()}, open("results/mengtijden.json", "w"))
        print("mengtijden t95:", tm)
    if "particles" in which:
        st = anim_particles(D)
        np.savez("results/deeltjes_stats.npz", **{k: np.array(v) for k, v in st.items()})
