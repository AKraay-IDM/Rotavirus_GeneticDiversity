"""
Do simple profiling
"""

import sciris as sc
import rotaABM as rabm

with sc.cprofile() as cpr:
    rota = rabm.RotaABM(N=10_000, timelimit=2)
    rota.run()

def run():
    rota = rabm.RotaABM(N=10_000, timelimit=2)
    rota.run()

prof = sc.profile(run=run, follow=rabm.RotaABM.contact_event)