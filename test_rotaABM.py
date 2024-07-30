"""
Check results against baseline values.

NB: the two tests could be combined into one, but are left separate for clarity.
"""

import sciris as sc
import rotaABM as rota
import rotaABM_IDM_VACCINE_age_experiment_ck as rota_orig


def test_default(make=False):
    sc.heading('Testing default parameters')
    filename = 'test_events_default.json'
    
    # Generate new baseline
    if make:
        events = rota_orig.main()
        sc.savejson(filename, events)
        
    # Check old baseline
    else:
        events = rota.main()
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, 'Events do not match for default simulation'
        print(f'Defaults matched:\n{events}')
        
    return


def test_alt(make=False):
    sc.heading('Testing alternate parameters')
    filename = 'test_events_alt.json'
    inputs = ['', # Placeholder (file name)
        2,   # immunity_hypothesis
        0.2, # reassortment_rate
        2,   # fitness_hypothesis
        2,   # vaccine_hypothesis
        2,   # experimentNumber
    ]
    
    # Generate new baseline
    if make:
        events = rota_orig.main(inputs)
        sc.savejson(filename, events)
        
    # Check old baseline
    else:
        events = rota.main()
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, 'Events do not match for alternate parameters simulation'
        print(f'Alternate parameters matched:\n{events}')
        
    return


if __name__ == '__main__':
    make = True # Set to True to regenerate results
    test_default(make=make)
    test_alt(make=make)