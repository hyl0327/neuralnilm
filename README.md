neuralnilmtp
==============

NeuralNILM TaiPower is an implementation of Jack
Kelly's [Neural NILM Paper](https://arxiv.org/abs/1507.06594) used by Taiwan
Power Company.

The data processing part of NeuralNILM TaiPower is mostly based
on [neuralnilm](https://github.com/JackKelly/neuralnilm), which is maintained by
the paper's author, while the training and prediction parts have been mostly
re-written. Most notably, Keras is being used in place of Theano for training.

Designed with scalability in mind, NeuralNILM comes with a pair of training and
prediction scripts that aim to be useful in an online learning situation, where
one can easily switch among different datasets or models for training and
prediction.


## Directories

The following parts of this documentation use `<ROOT_DIR>` to refer to
the project root directory (that is, the directory in which you cloned this
project), and `<*_DIR>` to refer to some other directories based on
`<ROOT_DIR>`.

Generally speaking, `<FOO_BAR_DIR>` refers to the directory
`<ROOT_DIR>/foo/bar/`. For complete information on directories, see
`<ROOT_DIR>/lib/dirs.py`.


## Datasets

NeuralNILM TaiPower expects datasets to be in the format of NILMTK
(http://nilmtk.github.io), which are basically `.h5` files, and requires them to
be placed in `<DATA_DIR>`.

For each dataset, one should add a corresponding configuration file to
`<CONFIG_DIR>`, which must have the same name as the dataset, but with an
extension of `.py` instead.

In this project, we use the REDD dataset as example (take a look at
`<DATA_DIR>/README.txt`), and an example configuration file `redd.py` is
included in `<CONFIG_DIR>`.

Finally, one should edit `<CONFIG_DIR>/seq_periods.py` to include the sequence
periods (refer to the paper) for the target appliances that are going to be
trained on. Note that `<CONFIG_DIR>/seq_periods.py` is shared among different
datasets.


## Models

NeuralNILM TaiPower saves the trained models in `<MODELS_DIR>`, which can be
used by prediction or further training. Models are distinguished by the
appliance types on which they have been trained.

On the other hand, `<TOPOLOGIES_DIR>` contains model topologies that specify how
the model should look like, in Keras' terms. In this project, we use a Denoising
Auto-encoder as the default model topology, which can be found as `dae.py`
located in `<TOPOLOGIES_DIR>`.


## Training

The training step is quite straightforward that an execution of
`<ROOT_DIR>/train.py -h` should give all the information one would need to know
about it.

After training, it will store its output in `<TRAIN_OUTPUT_DIR>`.


## Prediction

Prediction works by going though an arbitrarily long sequence of mains data, and
predicting a given appliance's energy usage along the way. It can optionally
take a validation sequence, which is the target energy usage (ground truth).

After prediction, it will store its output in `<PREDICTION_OUTPUT_DIR>`.
