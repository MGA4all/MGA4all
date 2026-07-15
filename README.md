# MGA4all
Various Modelling to Generate Alternative schemes for different energy system optimisation models

## Implemented Algorithms and supported backends

| Algorithm                                             | PyPSA + linopy |
|-------------------------------------------------------|----------------|
| [SPORES](https://doi.org/10.1016/j.joule.2020.08.002) | ✅             |
| [Random Directions][]                                 | ✅             |   

## Running MGA4all

We separate out the modelling backends into optional groups
(e.g. `pypsa`).  So depending on the model someone wants to work with,
you have to choose the appropriate backend.

We prefer using [`hatch`](https://hatch.pypa.io/latest/install/)
(>=1.16) to create/manage necessary environments and run commands

```
$ hatch run <command> [options]
```

Where, `<command>` can be any script that uses `MGA4all`; by default
the `pypsa` backend is used.

If you don't want to use `hatch`, create a virtual environment as you
would, install MGA4all in edit mode:

```
$ pip install -e .
```

and run your script as you normally would.

### Testing with included examples

MGA4All also includes a an example PyPSA model.  
A user can use this model for testing while working with MGA4All
interactively in a Python shell.

```python
import yaml

from mga4all.mga_random_directions import random_directions_algorithm
from mga4all.examples import create_pypsa_network


with open("configs/test_config_random_directions.yaml") as yf:
    test_config = yaml.safe_load(yf)

mynetwork = create_pypsa_network()
mynetwork.optimize(sovler_options={'solver_name': 'highs'})

mga_alternatives, mga_spatial_alternatives = random_directions_algorithm(test_config, mynetwork)
```
