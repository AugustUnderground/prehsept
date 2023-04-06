# prehsept

Modeling the behaviour of primitive devices around the operating point in
haskell.

## Setup

Requires [libtorch](https://pytorch.org/get-started/locally/) symlinked into
the root of this repository according to `extra-include-dirs` and
`extra-lib-dirs` in `stack.yaml`.

The `setenv` script can be sourced to export the correct `LD_LIBRARY_PATH`.
Additionally it exports the `DEVICE` variable, supposedly enabling GPU support
in Hasktorch (I'm not sure how/if it works).

```sh
$ source setenv
```

## Usage

Build the project with stack

```sh
$ stack build
```

then run it

```sh
$ stack run
```

### CLI

The executable part supports the following arguments:

```bash
Primitive Device Modeling Around the Operating Point

Usage: prehsept-exe [-k|--pdk PDK] [-d|--dev DEV] (-p|--dir DIR) [-n|--num NUM]
                    [-r|--reg REGION] [-s|--size SIZE]
  PREHSEPT

Available options:
  -k,--pdk PDK             PDK from which the data was generated
                           (default: xh035)
  -d,--dev DEV             Device Type: nmos | pmos (default: nmos)
  -p,--dir DIR             Path to lookup-table as tensor
  -n,--num NUM             Number of Epochs (default: 25)
  -r,--reg REGION          Region of Operation: 2 | 3 (default: 2)
  -s,--size SIZE           Batch Size (default: 5000)
  -h,--help                Show this help text
```

For example, to train a GPDK180 NMOS model for 100 epochs with a batch size of
32 run:

```bash
stack exec -- prehsept-exe --pdk gpdk180 --dev nmos --dir ./data/gpdk180-pmos.pt --size 32 --num 100
```

## Notebooks

The notebooks can be viewed locally by running the jupyter server in the
`./notebooks` directory:

```bash
$ stack exec jupyter -- notebook
```

[IHaskell](https://github.com/IHaskell/IHaskell) must be installed for this to
work.

## License

BSD3

## Thanks

Thanks to the [hastorch](https://github.com/hasktorch/hasktorch) project!
