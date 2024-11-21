import sciris as sc
import calibration.process_incidence as cpi

for i in [1,2]:
    hypo = 2 if i==1 else 1
    sc.heading(f'Debug {i}')
    df = sc.dataframe.read_csv(f'./debug{i}/results/rota_strains_infected_all_1_0.1_{hypo}_1_1_0_0.5_1.csv')
    out = cpi.process_model(df)
    print(out)