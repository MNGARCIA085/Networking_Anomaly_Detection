from sklearn.preprocessing import PowerTransformer, QuantileTransformer


TRANSFORM_REGISTRY = {
    "power": PowerTransformer,      # parametric distribution transform
    "quantile": QuantileTransformer # non-parametric distribution transform
}


def create_transform(name, **params):
    try:
        transform_cls = TRANSFORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown transform: {name}. "
            f"Available: {list(TRANSFORM_REGISTRY)}"
        )

    return transform_cls(**params)




"""
The two transformations test substantially different assumptions:

Yeo-Johnson / PowerTransformer: tries to make each feature's distribution more Gaussian 
through a parametric power transformation.
QuantileTransformer: uses the empirical distribution to map values to a target distribution. 
It can handle strongly skewed/heavy-tailed 
features without assuming a particular parametric shape.
"""


"""
# In the YAML

transform:
  enable: true
  name: quantile
  params:
    output_distribution: normal
    n_quantiles: 1000
"""