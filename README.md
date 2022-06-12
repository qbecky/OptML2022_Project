# OptML2022_Project

The project aims at evaluating the benefits of using [Closed Loop Control](https://proceedings.mlr.press/v119/xu20d.html) to stabilize GAN training. All the plots shown in the report can be obtained by running `notebooks/DiracGAN.ipynb`.

## Python environement

In order to obtain all the modules that are necessary, we suggest creating a virtual environment:

```bash
conda env create -n pytorch_env
conda activate pytorch_env
```

Then install PyTorch, matplotlib, jupyter lab, and Numpy.


## Running a jupyter notebook

To run a notebook, first launch jupyter lab in the one of the parent folders:

```
jupyter lab
```

Make sure you have activated the environment first. For instance, try running the notebook `notebooks/DiracGAN.ipynb`

