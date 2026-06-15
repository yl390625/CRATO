import os
import random

import numpy as np
from scipy.stats import uniform

# Create the folder if it does not exist
if not os.path.exists("demand_sample/N_samples_corr"):
    os.makedirs("demand_sample/N_samples_corr")

# Define the parameters of the uniform distribution
lower_bound = 0
upper_bound = 40
size_list =  [10, 100, 200]
dim = 2
prop = 0.9
# Create a uniform distribution object
uniform_dist = uniform(loc=lower_bound, scale=upper_bound - lower_bound)
for i in size_list:
    data = uniform_dist.rvs(size=(i, dim))

    tmp = random.random()
    if tmp<=prop:
        data[:,1]=40-data[:,0]

    # Save the data to a NumPy file in the demand_sample folder with the distribution information in the file name
    filename = f"demand_sample/N_samples_corr/num_samples_{i}.npy"
    np.save(filename, data)
