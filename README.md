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
                    [-s|--size SIZE]
  PREHSEPT

Available options:
  -k,--pdk PDK             PDK from which the data was generated (default: xh035)
  -d,--dev DEV             Device Type: nmos | pmos (default: nmos)
  -p,--dir DIR             Path to lookup-table as tensor
  -n,--num NUM             Number of Epochs (default: 25)
  -s,--size SIZE           Batch Size (default: 25)
  -h,--help                Show this help text
```

## Example Notebooks

soon.

## License

BSD3

## Thanks

Thanks to the [hastorch](https://github.com/hasktorch/hasktorch) project and
the stack [sekelton](https://github.com/hasktorch/hasktorch-stack-skeleton)!
