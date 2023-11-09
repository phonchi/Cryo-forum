# Cryo-forum -  A framework for orientation recovery with uncertainty measure with the application in cryo-EM image analysis

Cryo-forum is a framwork that allows the developers to experiment with different stratgies for orientation recovery in cryo-EM.

## Manuscript:

[Cryo-forum: A framework for orientation recovery with uncertainty measure with the application in cryo-EM image analysis](https://arxiv.org/abs/2307.09847)

Szu-Chi Chung

## How to use the library
### Install 

If you do not have the Anaconda python distribution, [please install it following the instructions](https://www.anaconda.com/download).

To create a conda environment for cryo-forum (More information on conda environments can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html)):
```
conda create --name cryoforum python=3.10
conda activate cryoforum
```

Now install the dependencies:
```
pip install mrcfile wandb ipyvolume seaborn tensorflow_probability==0.20.0 tensorflow==2.12 tensorflow-graphics tensorflow_similarity scikit-image ipython_genutils notebook quaternion
```

### Setup
Finally clone the repository and change the directory:
```
git clone https://github.com/phonchi/Cryo-forum.git
cd Cryo-forum
```

### Tutorials and guides
See the [Example Notebook](notebooks/) where you can find detail documentation about the analysis for synthetic and real datasets.

### Data and Pretrained model
If you wish to use the pretrained model or the data, you can find them in [pretrained model](/models) and [data](data/).

## License
Cryo-forum is open source software released under the [GNU General Public License, Version 3](https://github.com/phonchi/Cryo-forum/blob/master/LICENSE).

## Credit
The code base is uilt upon the code develpoed by [Jelena Banjac](https://github.com/JelenaBanjac/protein-reconstruction) and the code https://github.com/utiasSTARS/bingham-rotation-learning/tree/1c5ee6626f99522f85f1d6c106b1230be9be09cc.

The visulization code for binham distribution is adapted from [here](https://github.com/woven-planet/BinghamNLL/tree/main)
