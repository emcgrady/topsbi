## Configuration file fields

- `batchSize`: number of events to feed the network at at once
- `c0`: SM-inclusive set of WCs for label-0 events
- `c1`: SM-inclusive set of WCs for label-1 events (not used in parameterized training)
- `cg`: SM-inclusive set of WCs used to generate the sample
- `cr`: optional reference hypothesis to divide `c0` adn `c1` by 
- `data`: directory containing torch `TensorDatet`s with features in tuple index 0 and fit coefficients in tuple index 1. Training expects to see `train.p` and `test.p` in this directory for testing and training datasets, respectively
- `device`: The device used for training. Use `cpu` for CPU training and `cuda` for GPU training. For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- `epochs`: The number of epochs used to train.
- `learningRate`: Initial learning rate for network training
- `method`: 2 options
    1. `stitched`: choose a fixed `c1` to train with. WC values are latent information with respect to the network
    1. `parameterized`: scan values of `c1` for training. WC values are observable information with resepct to the network
- `name`: directory for output of training
- `network`: optional argument used for variable network architecture
- `ranges`: dictionary of WCs with ranges of WC values to randomly select from for parameterized training (not used in stitched training)
- `seed`: random seed used for network initialization
- `wcs`: array of WC names for sample