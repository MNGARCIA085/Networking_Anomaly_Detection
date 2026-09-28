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

	#print(X_train.shape)


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

	X_train_0, y_train_0, X_val_0, y_val_0 = prep.build_prep(
	    X_train,
	    y_train,
	    X_val,
	    y_val,
	)






	print("final tauin shape")
	print(X_train_0.shape)




	#print(X_train_0[:2])


	#return



	# print(X_train_0)



	# save artifacts
	prep.save("prep.joblib")



	# example of load
	from network_anomaly_detection.preprocessing.base import BasePrep
	prep_saved = BasePrep.load("prep.joblib")
	print(prep_saved)


	print(type(prep_saved))



	X_train_1, y_train_1 = (
	    prep.transform_with_labels(
	        X_train,
	        y_train,
	    )
	)

	X_val_1, y_val_1 = (
	    prep.transform_with_labels(
	        X_val,
	        y_val,
	    )
	)

	# reloaded prep
	X_train_2, y_train_2 = (
	    prep_saved.transform_with_labels(
	        X_train,
	        y_train,
	    )
	)

	X_val_2, y_val_2 = (
	    prep_saved.transform_with_labels(
	        X_val,
	        y_val,
	    )
	)



	assert np.array_equal(X_train_1, X_train_2)
	assert np.array_equal(y_train_1, y_train_2)
	assert np.array_equal(X_val_1, X_val_2)
	assert np.array_equal(y_val_1, y_val_2)

	print("✓ Saved and loaded prep produces identical results")


	


	#------

	print(
	    "pointwise:",
	    type(prep_saved.pointwise_prep)
	)

	print(
	    "windowing:",
	    type(prep_saved.windowing)
	)

	print(
	    "temporal_level:",
	    type(prep_saved.temporal_prep)
	)


	"""
	pointwise_prep, windowing, window_level_prep = (
	    prep.build_pipeline()
	)

	# Pointwise
	if pointwise_prep:
	    X_train = pointwise_prep.fit_transform(X_train)
	    X_val = pointwise_prep.transform(X_val)

	
	# Windowing
	if windowing:
		X_train = windowing.transform(X_train)

		X_val, y_val = windowing.transform_with_labels(
		    X_val,
		    y_val,
		)
	"""

	#print(X_train[:3])
	print(X_train.shape)
	print(y_train.shape)




	# adapt input es más del modelo, no del prep. general
	#X_train_model = entry.adapt_input(X_train)
	#X_val_model = entry.adapt_input(X_val)
	

	#print(X_val_model)
	#artifacts = prep.get_artifacts()


if __name__=="__main__":
	main()



"""
# Window-level
	if window_level_prep:
	    X_train = window_level_prep.fit_transform(X_train)
	    X_val = window_level_prep.transform(X_val)

		prep.fit(X_train)

		X_train_model = prep.transform(X_train)
		X_val_model, y_val_windowed = prep.transform(
		    X_val,
		    y_val,
		)
"""