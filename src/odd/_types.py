from dataclasses import dataclass

@dataclass(frozen=True)
class CoupledStripArrangement:
    V0: float # Potential of the sytem, used to scale the system which is defaulted at V0=1.0
    hw_arra: float # half width of the arrangement, parameter a
    ht_arra: float # height of the arrangement, parameter b
    ht_subs: float # height of the substrate, parameter h
    w_gap_strps: float # gap between the two microstrips, parameter s
    w_micrstr: float # width of the microstrip, parameter w
    ht_micrstr: float # height of the microstripm, parameter t
    er1: float # dielectric constatnt for medium 1
    er2: float # dielctric constant for medium 2
    num_fs: int # number of fourier series coefficients
    num_pts: int # number of points for the piece wise linear approaximation
    mode: str # Even or Odd mode