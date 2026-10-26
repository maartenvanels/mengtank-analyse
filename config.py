"""
Mengtank - centrale configuratie (alle aannames op één plek).

Methodiek-voorbeeld: 2D-axisymmetrische RANS (r, z + swirl) met een
blade-element impellermodel, gevolgd door FEM op de schoep en een
balkberekening van de as (koker) met de onderste driepoot-lagering.
"""
import numpy as np

# --- Medium -------------------------------------------------------------
RHO = 998.0            # kg/m3  water 20 C
MU = 1.0e-3            # Pa s
NU = MU / RHO

# --- Tank ---------------------------------------------------------------
T_DIAM = 4.0           # m   tankdiameter
H_LIQ = 7.2            # m   vloeistofhoogte = tankhoogte (tank volledig gevuld, gesloten deksel)

# --- As / koker -----------------------------------------------------------
SHAFT_LEN = 7.0        # m   aslengte (van aandrijving tot driepoot-lager)
SHAFT_OD = 0.324       # m   koker buitendiameter (DN300)
SHAFT_WALL = 0.010     # m   wanddikte koker
Z_BEARING = H_LIQ - SHAFT_LEN   # m  hoogte onderste (driepoot) lager = 0.2 m

# --- Roerder ------------------------------------------------------------
RPM = 100.0
N_REV = RPM / 60.0                 # 1/s
OMEGA = 2 * np.pi * N_REV          # rad/s
D_IMP = 1.5                        # m   schoepdiameter
R_IMP = D_IMP / 2
R_HUB = 0.22                       # m   schoepwortel (naaf op koker)
N_BLADES = 4                       # schoepen per waaier
BLADE_W = 0.30                     # m   schoepbreedte (koorde), W/D = 0.2
BLADE_T = 0.020                    # m   plaatdikte schoep (was 15 mm: voldeed niet op vermoeiing)
PITCH_DEG = 45.0                   # schoephoek, neerwaarts pompend
Z_IMPELLERS = [1.5, 4.5]           # m   hoogte waaiers (2 trappen op de 7 m as)

# Kalibratie: normaalkrachtcoeff. vlakke plaat C_N = K_CN * sin(alpha).
# K_CN is zo gekozen dat de vermogensgetal Np in de geschotte tank
# uitkomt op literatuurwaarde 4-blads PBT 45 graden (Np ~ 1.27).
K_CN = 3.25
NP_LIT = 1.27                      # literatuur (W/D=0.2, turbulent, geschot)
NQ_LIT = 0.79                      # literatuur flow number

# --- Keerschotten (baffles) ----------------------------------------------
N_BAFFLES = 4
BAFFLE_W = T_DIAM / 12             # m
CD_BAFFLE = 2.0

# --- Driepoot onderlager -------------------------------------------------
TRIPOD_LEGS = 3
TRIPOD_LEG_D = 0.114               # m  buisdiameter poot
TRIPOD_R = 0.80                    # m  voetradius poten op bodem
TRIPOD_H = Z_BEARING + 0.10        # m  hoogte driepoot-zone
CD_LEG = 1.2

# --- Numeriek -----------------------------------------------------------
DX = 0.04                          # m  celgrootte
L_MIX_MAX = 0.08                   # m  max. menglengte (Prandtl) turbulentiemodel
SC_T = 0.7                         # turbulent Schmidt-getal
CFL = 0.35

# --- Simulatieverloop -----------------------------------------------------
T_STARTUP = 240.0                  # s  opstart tot (quasi) stationair
T_MIX = 180.0                      # s  menging na injectie tracer
SNAP_DT = 0.5                      # s  interval opslag snapshots

# Tracer-injectie: blob onder het deksel nabij de wand (ongunstigste plek)
TRACER_BOX = dict(r=(1.40, 1.80), z=(6.60, 7.00))

# Monitorpunten (r, z) voor menggraad
PROBES = {
    "P1 bodem-midden": (0.40, 0.30),
    "P2 wand-onder": (1.85, 1.00),
    "P3 midden": (1.00, 3.00),
    "P4 wand-boven": (1.85, 6.50),
    "P5 deksel-as": (0.40, 7.00),
}

# --- Constructie (FEM) ----------------------------------------------------
E_STEEL = 200e9                    # Pa (RVS 316L)
NU_STEEL = 0.3
RHO_STEEL = 7950.0
SY_316L = 220e6                    # Pa  vloeigrens 316L (0.2%)
DYN_FACTOR = 1.5                   # dynamische / opstartfactor op hydraulische belasting
FLUCT_AMP = 0.30                   # +/- fluctuatie schoepbelasting t.o.v. gemiddelde (turbulentie, baffle-passage)
FAT_CLASS = 71e6                   # Pa  FAT71 (lasdetail schoepwortel, EN 1993-1-9)

CASES = {
    "geschot": dict(baffles=True, label="Met 4 keerschotten"),
    "ongeschot": dict(baffles=False, label="Zonder keerschotten"),
}
