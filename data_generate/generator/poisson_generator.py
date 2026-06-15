import os
import numpy as np
from scipy.stats import poisson

# Create the folder if it does not exist
if not os.path.exists("demand_sample/PC_samples"):
    os.makedirs("demand_sample/PC_samples")

# Define the lambda parameter of the Poisson distribution
lambdas1 = [1.2, 0.6, 1.2, 1]
size_list = [i*10 for i in range(1,101)]
for i in size_list:
    data = [poisson.rvs(mu=lambda_, size=i) for lambda_ in lambdas1]
    data = np.array(data).T  # Transpose the array to have samples as rows
    # Save the data to a NumPy file in the demand_sample folder with the distribution information in the file name
    filename = f"demand_sample/PC_samples/num_samples_{i}.npy"
    np.save(filename, data)