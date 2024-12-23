import h5py

# Load the HDF5 file
file_path = "./data/CAMUS/camus.h5"
with h5py.File(file_path, "r") as h5_file:
    # List all groups/datasets in the file
    print("Keys in the file:", list(h5_file.keys()))
    
    # Access a specific group or dataset
    dataset = h5_file["1"]  # Replace with the dataset name you want to access
    # print("Shape of the dataset:", dataset.shape)
    # print("Data type of the dataset:", dataset.dtype)
    breakpoint()
    print(h5_file["1"].keys())