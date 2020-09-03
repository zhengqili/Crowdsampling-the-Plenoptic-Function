from data.aligned_data_loader import *#, TestDataLoader, TUMDataLoader

def CreateLandmarksDataLoader(opt, data_dir, phase, num_threads, img_a_name=None, img_b_name=None, img_c_name=None):
    data_loader = LandmarksDataLoader(opt, data_dir, phase, num_threads, img_a_name, img_b_name, img_c_name) #CustomDatasetDataLoader()
    return data_loader
