"""Overzichtsfiguur tijdreeksen: vermogensgetal, pompgetal, swirl en menggraad."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams.update({"font.size": 9})
fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for k, ls in (("geschot", "-"), ("ongeschot", "--")):
    d = np.load(f"results/cfd_{k}.npz")
    t = d["h_t"]
    lab = C.CASES[k]["label"]
    m = t > 1.0
    axs[0, 0].plot(t[m], d["Np"][m], ls, lw=2, label=lab)
    axs[0, 1].plot(t[m], d["Nq"][m], ls, lw=2, label=lab)
    axs[1, 0].plot(t, d["swirl"], ls, lw=2, label=lab)
    mm = t >= d["t_inj"]
    axs[1, 1].semilogy(t[mm] - d["t_inj"], d["cov"][mm], ls, lw=2, label=lab)
axs[0, 0].axhline(C.NP_LIT, color="k", ls=":", lw=1); axs[0, 0].text(250, C.NP_LIT + 0.04, "literatuur Np = 1.27 (geschot)")
axs[0, 1].axhline(C.NQ_LIT, color="k", ls=":", lw=1); axs[0, 1].text(250, C.NQ_LIT + 0.03, "literatuur Nq = 0.79")
axs[1, 1].axhline(0.05, color="k", ls=":", lw=1); axs[1, 1].text(100, 0.056, "95% homogeen")
axs[0, 0].set(title="Vermogensgetal Np per waaier", xlabel="tijd [s]", ylim=(0, 2.5))
axs[0, 1].set(title="Pompgetal Nq = Q / (N D³)", xlabel="tijd [s]", ylim=(0, 1.2))
axs[1, 0].set(title="Gemiddelde swirl  v$_\\theta$ / v$_{tip}$ (meedraaien inhoud)", xlabel="tijd [s]")
axs[1, 1].set(title="Menggraad na tracer-injectie (CoV)", xlabel="tijd na injectie [s]", ylim=(1e-3, 30))
for ax in axs.flat:
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.savefig("animaties/tijdreeksen.png", dpi=110)
