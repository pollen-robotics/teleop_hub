import os

from ruamel.yaml import YAML  # type: ignore


def update_config(path: str, updates: dict):
    """
    Update a YAML config file (preserving comments and order) with new values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(path, "r") as file:
        config = yaml.load(file)

    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    updated_config = recursive_update(config, updates)

    with open(path, "w") as file:
        yaml.dump(updated_config, file)

    print(f"Config updated at {path}")
