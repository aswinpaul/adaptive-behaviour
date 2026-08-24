# -*- coding: utf-8 -*-
import pandas as pd

from adaptive_behaviour import paths

#: Per-seed output written by the trial runner.
RAW = paths.results_dpefe_raw(10)
#: Where the concatenated result is written.
OUT = paths.RESULTS_DPEFE

m_trials = 99

p_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'data_' + str(i) +'.csv'
    data = pd.read_csv(RAW / file_name)
    p_data = pd.concat([p_data, data])

a_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'a_data_' + str(i) +'.csv'
    data = pd.read_csv(RAW / file_name)
    a_data = pd.concat([a_data, data])
    
b_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'b_data_' + str(i) +'.csv'
    data = pd.read_csv(RAW / file_name)
    b_data = pd.concat([b_data, data])
    
c_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'c_data_' + str(i) +'.csv'
    data = pd.read_csv(RAW / file_name)
    c_data = pd.concat([c_data, data])
    
p_data.to_csv(OUT / 'p_data_10.csv', index=False)
a_data.to_csv(OUT / 'a_data_10.csv', index=False)
b_data.to_csv(OUT / 'b_data_10.csv', index=False)
c_data.to_csv(OUT / 'c_data_10.csv', index=False)
