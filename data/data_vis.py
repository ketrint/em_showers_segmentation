import pandas as pd
import numpy as np

data = pd.read_csv('showers_18k_final_ver1.csv')


print(data.head(2))


import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
rcParams['axes.labelpad'] = 8.0

sns.set(context='paper', style="whitegrid", font_scale=2.5, font = 'serif')

linewidth = 3

plt.figure(figsize=(10, 6), dpi=100)
plt.hist(data.ele_P, label = 'mean energy = '+str(np.round(np.mean(data.ele_P),1)), bins =100, color = 'firebrick', alpha = 0.7)
plt.xlabel('Energy, GeV')
plt.legend()
plt.savefig("E_true_distr.pdf", bbox_inches='tight')

min_z = np.abs(data['ele_SZ'].min())
z = data['SZ'] + min_z
z_0 = data['ele_SZ'] + min_z

plt.figure(figsize=(10, 6), dpi=100)
plt.hist(z, label = 'mean $z$ coordinate = '+str(np.round(np.mean(z),1)), bins =100, alpha = 0.7)
plt.xlabel('$z$, $\mu m$')
plt.legend()
plt.savefig("z_distr.pdf", bbox_inches='tight')

plt.figure(figsize=(10, 6), dpi=100)
plt.hist(z_0, label = 'mean $z_0$ coordinate = '+str(np.round(np.mean(z_0),1)), bins =100, alpha = 0.7)
plt.xlabel('$z_0$, $\mu m$')
plt.legend()
plt.savefig("z_0_distr.pdf", bbox_inches='tight')

