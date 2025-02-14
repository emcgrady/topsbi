## Configuration file fields

- `backgroundMean`: The mean of the weights used for background events. In the current iteration of the training, this is used to normalize the weights of the background events before training.
- `backgroundMean`: The directory of the `features.p` and `fit_coefs.p` files for the background events. `features.p` should contain an $m$ by $n$ torch tensor where $m$ is the number of events and $n$ is the number of features. Similarly, `fit_coefs.p` should be an $m$ by $p$ torch tensor where $p$ is the number of terms of your SMEFT quadratic expansion.
- `backgroundStartingPoint`: SMEFT starting point used to generate the background events.[^1]
- `backgroundTrainingPoint`: The point in WC space to reweight the background sample to before training.[^1]
- `batchSize`: The sample size used for training
- `device`: The device used for training. Use `cpu` for CPU training and `cuda` for GPU training. For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- `epochs`: The number of epochs used to train.
- `factor`: This network uses `ReduceLROnPlateau` to adjust the learning rate upon a plateau. This argument is how much the learning rate is adjuted.
- `featureMeans`: The means of the input features which is used to normalize before training.
- `featureStdvs`: The features standard deviations which is used to normalize before training.
- `features`: A list of features used for training.
- `learningRate`: Initial learning rate for network training
- `name`: Directory used for plots and output networks. Currently, these are saved every 50 epochs.
- `patience`: This network uses `ReduceLROnPlateau` to adjust the learning rate upon a plateau. This is the number of epochs where no improvemen is detected before the learning rate is adjusted.
- `signalMean`: The mean of the weights used for signal events. In the current iteration of the training, this is used to normalize the weights of the signal events before training.
- `signalSample`: The directory of the `features.p` and `fit_coefs.p` files for the signal events. `features.p` should contain an $m$ by $n$ torch tensor where $m$ is the number of events and $n$ is the number of features. Similarly, `fit_coefs.p` should be an $m$ by $p$ torch tensor where $p$ is the number of terms of your SMEFT quadratic expansion.
- `signalStartingPoint`: SMEFT starting point used to generate the signal events. [^1]
- `signalTrainingPoint`: Point in WC space used to train non-quadratic terms. Use `null` if training a quadratic term. [^1]
- `signalTrainingTerm`: Quadratic term to train on. Use `null` if training a non-quadratic term.

[^1]: The syntax for a point in WC space is the name of a term and its WC value connected via an equal sign. Use a colon to separate multiple terms. (i.e. ctq8=2.6:ctq1=1.6)
