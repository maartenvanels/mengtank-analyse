# Mengtank Ø4 m – CFD en FEM

Methodiek-voorbeeld: stroming, menging en sterkte van een mengtank (Ø4 m, 7,2 m hoog, gevuld met water)
met een dubbele 45°-roerder (Ø1,5 m, 100 omw/min) op een koker van 7 m met een driepoot-onderlager.

**Rapport met animaties:** zie de GitHub Pages-site van deze repository (`index.html`).

## Rekenketen

| Stap | Script | Wat |
|---|---|---|
| 1 | `cfd_axisym.py` | 2D-axisymmetrische RANS-stromingsberekening met schoepelement-model, tracer-menging |
| 2 | `fem_blade.py` | FEM van de schoep (scikit-fem, kwadratische tetraëders), spanning, vermoeiing, eigenfrequenties |
| 3 | `shaft_beam.py` | Balkmodel van de koker met driepoot-lager, kritisch toerental |
| 4 | `fig_summary.py`, `anim_cfd.py`, `anim_fem.py` | Figuren en animaties (map `animaties/`) |

Alle aannames staan in `config.py`.

## Opnieuw draaien

Vereist Python 3.12 met numpy, scipy, matplotlib, numba, scikit-fem en ffmpeg.

```
python cfd_axisym.py geschot
python cfd_axisym.py ongeschot
python fem_blade.py
python shaft_beam.py
python fig_summary.py
python anim_cfd.py
python anim_fem.py
```

De map `results/` (±200 MB) staat niet in de repository; die wordt door de scripts aangemaakt.
