import os
import pickle5 as pickle

def load_STT():
	this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
	data_path = os.path.join(this_dir, 'STTm.pkl')
	print(data_path)
	with open(data_path, "rb") as fh:
	  data = pickle.load(fh)
	return data