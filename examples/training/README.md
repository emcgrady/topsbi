## Configuration file fields

- `backgroundSample`: The directory of the `features.p` and `fit_coefs.p` files for the background events. `features.p` should contain an $m$ by $n$ torch tensor where $m$ is the number of events and $n$ is the number of features. Similarly, `fit_coefs.p` should be an $m$ by $p$ torch tensor where $p$ is the number of terms of your SMEFT quadratic expansion.
- `backgroundStartingPoint`: List of SMEFT starting point used to generate the background events. 
- `backgroundTrainingPoint`: List of WCs used to reweight the background sample to before training.
- `batchSize`: The sample size used for training
- `device`: The device used for training. Use `cpu` for CPU training and `cuda` for GPU training. For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- `epochs`: The number of epochs used to train.
- `factor`: This network uses `ReduceLROnPlateau` to adjust the learning rate upon a plateau. This argument is how much the learning rate is adjuted.
- `learningRate`: Initial learning rate for network training
- `name`: Directory used for plots and output networks. Currently, these are saved every 50 epochs.
- `patience`: This network uses `ReduceLROnPlateau` to adjust the learning rate upon a plateau. This is the number of epochs where no improvemen is detected before the learning rate is adjusted.
- `signalSample`: The directory of the `features.p` and `fit_coefs.p` files for the signal events. `features.p` should contain an $m$ by $n$ torch tensor where $m$ is the number of events and $n$ is the number of features. Similarly, `fit_coefs.p` should be an $m$ by $p$ torch tensor where $p$ is the number of terms of your SMEFT quadratic expansion.
- `signalStartingPoint`: List of SMEFT starting point used to generate the signal events. 
- `signalTrainingPoint`: PList of WCs used to reweight the signal sample to before training.