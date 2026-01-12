# MGA4all
Various Modelling to Generate Alternative schemes for different energy system optimisation models

## Implemented Algorithms and supported backends

| Algorithm | PyPSA + linopy |
|-----------|----------------|
|[SPORES](https://doi.org/10.1016/j.joule.2020.08.002) | ✅ |

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
