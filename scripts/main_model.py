import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.data.data_module import DataModule







@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
	#...
	data = DataModule(
    	"data/arriba.csv",
	    "data/arriba.csv",
	)	

	X_train, y_train, X_val, y_val = data.load()

	#print (X_train[:3])
	#print(y_train[:3])

	print(X_train.shape)
	print(X_val.shape)


	# prep
	import numpy as np
	from network_anomaly_detection.models.nn.ae.prep import AEPrep



	#prep = AEPrep(cfg.model_type)
	from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY

	#print(cfg.model_type.name)


	prep_cls = PREP_REGISTRY[cfg.model_type.name]
	prep = prep_cls(cfg.model_type)

	#print(prep)




	#print("MAIN:", X_train.shape)

	X_train, y_train, X_val, y_val = prep.build_prep(
	    X_train,
	    y_train,
	    X_val,
	    y_val,
	)


	# models test
	from network_anomaly_detection.models.registry import MODEL_REGISTRY
	model_cls = MODEL_REGISTRY[cfg.model_type.name]
	model = model_cls(cfg.model_type.models)


	print(model)


	X_train = model.adapt_input(X_train)
	X_val = model.adapt_input(X_val)

	#
	print('final shapes')
	print(X_train.shape)
	print(X_val.shape)







if __name__=="__main__":
	main()


