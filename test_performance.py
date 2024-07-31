"""
Check performance
"""

import sciris as sc
import rotaABM as rota


def update_performance(save=False):
    sc.heading('Updating performance')
    filename = 'test_performance.json'

    T = sc.timer()
    rota.main()
    T.toc()
    string = f'{T.elapsed:0.2f}'
    data = dict(time=string)
    if save:
        sc.savejson(filename, data)
    print(data)
        
    return


def profile():
    sc.heading('Running sc.profile')
    prf = sc.profile(rota.main)
    return prf


def cprofile():
    sc.heading('Running sc.cprofile')
    with sc.cprofile() as cpr:
        rota.main(verbose=False)
    return cpr


if __name__ == '__main__':
    save = 0
    do_profile = 1
    if not do_profile:
        update_performance(save=save)
    else:
        prf = profile()
        cpr = cprofile()