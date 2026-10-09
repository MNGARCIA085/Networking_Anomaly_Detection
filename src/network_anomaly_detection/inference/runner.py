

class InferenceRunner:

    def __init__(
        self,
        prep,
        model,
        thresholding=None,
    ):
        self.prep = prep
        self.model = model
        self.thresholding = thresholding


    def _transform(self, X):
        X_p = self.prep.transform(X)
        return self.model.adapt_input(X_p)


    def _transform_with_labels(self, X, y):
        X_p, y_p = self.prep.transform_with_labels(X, y)
        return self.model.adapt_input(X_p), y_p


    def score(self, X):
        X_model = self._transform(X)
        return self.model.score(X_model)


    def predict(self, X):
        X_model = self._transform(X)

        threshold = (
            self.thresholding.get_threshold()
            if self.thresholding is not None
            else None
        )

        return self.model.predict(
            X_model,
            threshold=threshold,
        )

    def predict_with_labels(self, X, y):
        X_model, y_processed = self._transform_with_labels(X, y)

        scores = self.model.score(X_model)

        threshold = (
            self.thresholding.get_threshold()
            if self.thresholding is not None
            else None
        )

        predictions = self.model.predict(
            X_model,
            threshold=threshold,
        )

        return scores, y_processed, predictions





"""
Preprocessing: prep.transform() owns pointwise preprocessing, windowing, and temporal 
preprocessing. The runner should not reproduce those steps.

Input adaptation: model.adapt_input() handles model-specific shape requirements, such as 
flattening windows for an autoencoder or preserving 3D sequences for another architecture.

Thresholding: when no thresholder is loaded, threshold=None is passed to model.predict(). 
Your model's prediction interface must support that case.

Labels: prep.transform_with_labels() must preserve the alignment between transformed samples 
and their labels, especially when windowing changes the sample count.
"""








"""
class InferenceRunner:

    def __init__(
        self,
        prep,
        windowing,
        entry,
        wrapper,
        temporal_prep=None,
        thresholding=None,
    ):
        self.prep = prep
        self.temporal_prep = temporal_prep
        self.windowing = windowing
        self.entry = entry
        self.wrapper = wrapper
        self.thresholding = thresholding

    def _transform(self, X):

        X_p = self.prep.transform(X)

        X_w = self.windowing.transform(X_p)

        if self.temporal_prep is not None:
            X_w = self.temporal_prep.transform(X_w)

        return self.entry.adapt_input(X_w)


    def _transform_with_labels(self, X, y):

        X_p = self.prep.transform(X)

        X_w, y_w = self.windowing.transform_with_labels(
            X_p,
            y,
        )

        if self.temporal_prep is not None:
            X_w = self.temporal_prep.transform(X_w)

        X_model = self.entry.adapt_input(X_w)

        return X_model, y_w





    def score(self, X):

        X_model = self._transform(X)

        return self.wrapper.get_scores(X_model)





    def predict(self, X):

        X_model = self._transform(X)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            return self.wrapper.predict(
                X_model,
                threshold,
            )

        return self.wrapper.predict(X_model)




    def predict_with_labels(self, X, y):

        X_model, y_w = self._transform_with_labels(
            X,
            y,
        )

        scores = self.wrapper.get_scores(X_model)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            predictions = self.wrapper.predict(
                X_model,
                threshold,
            )

        else:

            predictions = self.wrapper.predict(
                X_model
            )

        return scores, y_w, predictions

"""











"""
Raw X
 ↓
prep.transform()
 ↓
windowing.transform()
 ↓
window_level_prep.transform()   # optional
 ↓
entry.adapt_input()
 ↓
wrapper.predict()
"""