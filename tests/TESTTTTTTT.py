import numpy as np
import scipy.io

# RANDOM
mat_data  = scipy.io.loadmat('./refTraj.mat')

# Extract the 'refTraj' structured array
refTraj = mat_data['refTraj']
# Access the fields within the structured array
referenceStates = refTraj['y'][0, 0]        # Main trajectory data


# Display a sample from each field
print("Sample of Trajectory Data (y):", referenceStates[:, 0])  # Show first 5 rows
