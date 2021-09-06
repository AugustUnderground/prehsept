# prehsept

[PRECEPT](https://github.com/electronics-and-drives/precept) but in haskell.

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

and run

```sh
$ stack run
```

## TODO

- [ ] Add argument parser for training data path etc.
- [ ] Add Tests.
- [X] Try data loaders. (Not working as expected)
- [ ] Move training loop to Lib.

## License

BSD3

## Thanks

Thanks to the [hastorch](https://github.com/hasktorch/hasktorch) project and
the stack [sekelton](https://github.com/hasktorch/hasktorch-stack-skeleton)!
