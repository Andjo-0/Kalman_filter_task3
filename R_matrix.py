import pandas as pd
import numpy as np

df = pd.read_csv("Data_collection/variances/output_wrench_log.csv")

# Only keep the wrench columns
wrenches = df[["Fx", "Fy", "Fz", "Mx", "My", "Mz"]]

R = wrenches.cov()
print(R)

R_mat = R.to_numpy()

np.savetxt("Data_collection/R/R_matrix.csv", R_mat, delimiter=",")




