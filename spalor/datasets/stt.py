import os
import pickle5 as pickle

def Nielsen2002():
	this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
	data_path = os.path.join(this_dir, 'STTm.pkl')
	with open(data_path, "rb") as fh:
	  data = pickle.load(fh)

	genes=pickle.load(open(os.path.join(this_dir, 'STTa.pkl'), 'rb')).query("~(Gene == '') ")['Gene']
	data=data.iloc[genes.index]
	data.index=genes
	return data