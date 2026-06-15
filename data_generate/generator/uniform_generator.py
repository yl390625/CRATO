import os
import numpy as np
from scipy.stats import uniform

# Create the folder if it does not exist
if not os.path.exists("demand_sample/N_samples"):
    os.makedirs("demand_sample/N_samples")

# Define the parameters of the uniform distribution
lower_bound = 0
upper_bound = 40
size_list = [10, 100, 200]
dim = 2
# Create a uniform distribution object
uniform_dist = uniform(loc=lower_bound, scale=upper_bound - lower_bound)


for i in size_list:
    # Generate 1000 data points from the uniform distribution
    data = np.concatenate((uniform_dist.rvs(size=(i, 1)),uniform_dist.rvs(size=(i, 1))),axis=1)

    # Save the data to a NumPy file in the demand_sample folder with the distribution information in the file name
    filename = f"demand_sample/N_samples/num_samples_{i}.npy"
    np.save(filename, data)
