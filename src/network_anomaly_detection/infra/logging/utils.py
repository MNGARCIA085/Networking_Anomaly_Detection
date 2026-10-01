




from collections.abc import Mapping

def flatten_dict(d, parent_key=""):
    items = {}

    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, Mapping):
            items.update(flatten_dict(v, key))

        elif isinstance(v, list):
            items[key] = str(v)

        else:
            items[key] = v

    return items







# helper
def flatten_dictv0(d, parent_key=""):
    items = {}

    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(
                flatten_dict(v, key)
            )

        elif isinstance(v, list):
            items[key] = str(v)

        else:
            items[key] = v

    return items