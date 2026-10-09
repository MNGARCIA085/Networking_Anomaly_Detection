import joblib




class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps


    """
    def fit(self, X):
        for step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X
    """

    def fit(self, X):
        for step in self.steps:
            print(f"FIT  {type(step).__name__}: {type(X).__name__}")
            step.fit(X)
            X = step.transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            print(f"TRANSFORM  {type(step).__name__}: {type(X).__name__}")
            X = step.transform(X)
        return X

    def fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X



    """ maybe later with 1x1
    # save artifact
    def save(self, path):

        joblib.dump(
            self,
            path
        )
    """
    




# check code later!!!!; data leakage......