"""
Animaties FEM:
  1. schoep_spanning.mp4   - waaier in 3D, von Mises (ontwerpbelasting) + overdreven vervorming,
                              fluctuerende belasting, draaiende camera
  2. schoep_eigenmodes.mp4 - eerste vier eigenvormen van de schoep (droog / nat vs. BPF)
  3. as_driepoot.png       - doorbuiging en moment van de koker, driepoot-lagerreactie
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import config as C

OUT = "animaties"
plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})
B = np.radians(C.PITCH_DEG)
W, T = C.BLADE_W, C.BLADE_T


def blade_xyz(xs, ys, ux, uy, uz, scale, phi, zc=0.0, zl=T / 2):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x = X + scale * ux
    y = Y + scale * uy
    z = zl + scale * uz
    r = C.R_HUB + x
    s = W * np.cos(B) / 2 - y * np.cos(B) + z * np.sin(B)
    a = zc + W * np.sin(B) / 2 - y * np.sin(B) - z * np.cos(B)
    return r * np.cos(phi) - s * np.sin(phi), r * np.sin(phi) + s * np.cos(phi), a


def anim_stress(F, fps=20, nfr=120, scale=15.0):
    xs, ys = F["xs"], F["ys"]
    vm = F["vm_grid"] * C.DYN_FACTOR / 1e6
    vmax = 130.0   # vaste schaal, zodat 15 mm en 20 mm vergelijkbaar zijn
    cmap = plt.cm.turbo
    norm = plt.Normalize(0, vmax)
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_axes([0.0, 0.05, 0.58, 0.9], projection="3d")
    ax2 = fig.add_axes([0.63, 0.55, 0.33, 0.33])
    ax3 = fig.add_axes([0.63, 0.12, 0.33, 0.30])
    cax = fig.add_axes([0.06, 0.08, 0.4, 0.02])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                 label=f"von Mises [MPa]  (ontwerpbelasting = {C.DYN_FACTOR} x gemiddeld)")

    # static 2D panels
    X, Y = np.meshgrid(xs + C.R_HUB, ys, indexing="ij")
    pc = ax2.contourf(X, Y * 1e3, vm, levels=24, cmap=cmap, norm=norm)
    ax2.set_xlabel("radius [m]"); ax2.set_ylabel("koorde [mm]  (0 = voorrand)")
    ax2.set_title("Spanningsverdeling schoep (bovenaanzicht)")
    ax2.annotate("las / inklemming", (C.R_HUB, W * 500), (C.R_HUB + 0.12, W * 850),
                 arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax3.plot(F["xq"] + C.R_HUB, F["qq"] / 1e3, "C3-", lw=2)
    ax3.fill_between(F["xq"] + C.R_HUB, F["qq"] / 1e3 * (1 - C.FLUCT_AMP), F["qq"] / 1e3 * (1 + C.FLUCT_AMP),
                     color="C3", alpha=0.2, label=f"+/- {int(C.FLUCT_AMP*100)}% fluctuatie")
    ax3.set_xlabel("radius [m]"); ax3.set_ylabel("normaalkracht [kN/m]")
    ax3.set_title(f"Belasting uit CFD (waaier {int(F['imp_index'])+1}), per schoep: {F['F_tot']/1e3:.2f} kN")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    def draw(fi):
        ax.clear()
        lamb = 1 + C.FLUCT_AMP * np.sin(2 * np.pi * fi / 30)
        rot = 2 * np.pi * fi / nfr * 0.25
        for b in range(C.N_BLADES):
            phi = rot + 2 * np.pi * b / C.N_BLADES
            for zl in (T / 2, -T / 2):
                Xg, Yg, Zg = blade_xyz(xs, ys, F["ux"] * lamb, F["uy"] * lamb, F["uz"] * lamb, scale, phi, zl=zl)
                ax.plot_surface(Xg, Yg, Zg, facecolors=cmap(norm(vm * lamb)), linewidth=0, antialiased=False,
                                shade=False, rstride=1, cstride=1)
            # undeformed outline
            X0, Y0, Z0 = blade_xyz(xs, ys, 0 * F["ux"], 0 * F["uy"], 0 * F["uz"], 0, phi)
            for e in (0, -1):
                ax.plot(X0[e, :], Y0[e, :], Z0[e, :], "k-", lw=0.4, alpha=0.5)
                ax.plot(X0[:, e], Y0[:, e], Z0[:, e], "k-", lw=0.4, alpha=0.5)
        # shaft + hub
        th = np.linspace(0, 2 * np.pi, 40)
        for zz in (-0.4, 0.4):
            ax.plot(C.SHAFT_OD / 2 * np.cos(th), C.SHAFT_OD / 2 * np.sin(th), zz, color="0.3", lw=1)
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.plot([C.SHAFT_OD / 2 * np.cos(a)] * 2, [C.SHAFT_OD / 2 * np.sin(a)] * 2, [-0.4, 0.4], color="0.5", lw=0.8)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.4, 0.4)
        ax.set_box_aspect((1.6, 1.6, 0.8))
        ax.view_init(elev=28, azim=-50 + 360 * fi / nfr)
        ax.set_axis_off()
        ax.set_title(f"Schoepspanning, vervorming x{scale:.0f} overdreven\n"
                     f"belasting {lamb*100:5.1f}% van gemiddeld   |   max {vm.max()*lamb:5.0f} MPa "
                     f"(vloeigrens 316L {C.SY_316L/1e6:.0f} MPa)", fontsize=10)
        return []

    draw(0)
    fig.savefig(f"{OUT}/schoep_spanning.png", dpi=110)
    anim = FuncAnimation(fig, draw, frames=nfr, blit=False)
    anim.save(f"{OUT}/schoep_spanning.mp4", writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=90)
    plt.close(fig)


def anim_modes(F, fps=20, nfr=60):
    xs, ys = F["xs"], F["ys"]
    fig = plt.figure(figsize=(12, 8))
    axs = [fig.add_subplot(2, 2, k + 1, projection="3d") for k in range(4)]
    X, Y = np.meshgrid(xs + C.R_HUB, ys, indexing="ij")

    def draw(fi):
        s = np.sin(2 * np.pi * fi / nfr)
        for k, ax in enumerate(axs):
            ax.clear()
            uz, uy = F["mode_uz"][k], F["mode_uy"][k]
            amp = np.abs(np.concatenate([uz.ravel(), uy.ravel()])).max()
            Zs = 0.08 * s * uz / amp
            Ys = Y + 0.08 * s * uy / amp
            ax.plot_surface(X, Ys, Zs, facecolors=plt.cm.coolwarm(0.5 + 0.5 * s * uz / amp), shade=False, lw=0)
            ax.set_zlim(-0.1, 0.1); ax.set_box_aspect((2, 1.2, 0.6))
            ax.set_axis_off()
            typ = "uit vlak" if F["zfrac"][k] > 0.5 else "in vlak"
            ax.set_title(f"Mode {k+1} ({typ}): droog {F['f_dry'][k]:.0f} Hz, nat ~{F['f_wet'][k]:.0f} Hz",
                         fontsize=10)
            ax.view_init(elev=22, azim=-60)
        fig.suptitle(f"Eigenvormen schoep  -  aanstoting: toerental {C.N_REV:.2f} Hz, "
                     f"schoep-/keerschotpassage {F['f_bpf']:.2f} Hz", fontsize=12, fontweight="bold")
        return []

    draw(nfr // 4)
    fig.savefig(f"{OUT}/schoep_eigenmodes.png", dpi=100)
    anim = FuncAnimation(fig, draw, frames=nfr, blit=False)
    anim.save(f"{OUT}/schoep_eigenmodes.mp4", writer=FFMpegWriter(fps=fps, bitrate=2500), dpi=85)
    plt.close(fig)


def plot_shaft(S):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    z = S["z"]
    axs[0].plot(S["defl"] * 1e3, z, lw=2)
    axs[0].set_xlabel("doorbuiging [mm]"); axs[0].set_ylabel("hoogte [m]")
    axs[0].set_title("Koker: doorbuiging (ontwerp)")
    axs[1].plot(S["Mb"] / 1e3, z, "C1", lw=2)
    axs[1].set_xlabel("buigmoment [kNm]"); axs[1].set_title("Buigmoment")
    for ax in axs[:2]:
        for zi in C.Z_IMPELLERS:
            ax.axhline(zi, color="0.6", ls="--", lw=0.8)
        ax.axhline(C.Z_BEARING, color="k", lw=1)
        ax.grid(alpha=0.3); ax.set_ylim(0, C.H_LIQ)
    ax = axs[2]
    ax.axis("off")
    txt = (f"As / koker  Ø{C.SHAFT_OD*1e3:.0f} x {C.SHAFT_WALL*1e3:.0f} mm, lengte {C.SHAFT_LEN} m\n"
           f"boven ingeklemd (aandrijving), onder driepoot-steunlager\n\n"
           f"koppel per waaier          {S['torque_imp']/1e3:6.2f} kNm\n"
           f"radiale hydr. kracht        {S['F_r']/1e3:6.2f} kN / waaier\n"
           f"max doorbuiging             {np.abs(S['defl']).max()*1e3:6.2f} mm\n"
           f"buigspanning                {S['sig_b']/1e6:6.1f} MPa\n"
           f"torsiespanning              {S['tau_t']/1e6:6.1f} MPa\n"
           f"von Mises                   {S['sig_vm']/1e6:6.1f} MPa\n\n"
           f"reactie driepootlager       {abs(S['R_bottom'])/1e3:6.2f} kN\n"
           f"max pootkracht (hoek {S['leg_phi_deg']:.0f}°)  {S['leg_max']/1e3:6.2f} kN\n\n"
           f"1e eigenfreq. droog          {S['f_droog'][0]:6.2f} Hz\n"
           f"1e eigenfreq. nat            {S['f_nat'][0]:6.2f} Hz\n"
           f"toerental                    {C.N_REV:6.2f} Hz\n"
           f"verhouding (nat/toerental)   {S['f_nat'][0]/C.N_REV:6.1f}  (eis > 1.3)")
    ax.text(0, 0.98, txt, va="top", family="monospace", fontsize=9.5)
    fig.savefig(f"{OUT}/as_driepoot.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    F = dict(np.load("results/fem_blade.npz"))
    anim_stress(F)
    anim_modes(F)
    plot_shaft(dict(np.load("results/shaft.npz")))
