# Cryo-forum
**A Framework for Orientation Recovery with Uncertainty Measurement in Cryo-EM Image Analysis**

Cryo-forum is a framework designed to facilitate experimentation with various strategies for orientation recovery in cryo-electron microscopy (cryo-EM).

## Manuscript

- **Cryo-forum: A Framework for Orientation Recovery with Uncertainty Measurement in Cryo-EM Image Analysis**
  - Author: Szu-Chi Chung
  - [Read the Manuscript](https://arxiv.org/abs/2307.09847)

## How to Use the Library

### Installation 

**Prerequisite**: If you do not have the Anaconda Python distribution, [install it here](https://www.anaconda.com/download).

**Create a Conda Environment for Cryo-forum**: 
- More information on managing conda environments can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).
```
conda create --name cryoforum python=3.10
conda activate cryoforum
```

**Install Dependencies**:
```
pip install mrcfile wandb ipyvolume seaborn tensorflow_probability==0.20.0 tensorflow==2.12 tensorflow-graphics tensorflow_similarity scikit-image ipython_genutils notebook quaternion
```

### Setup

**Clone the Repository and Change Directory**:
```
git clone https://github.com/phonchi/Cryo-forum.git
cd Cryo-forum
```

### Tutorials and Guides

- Explore the [Example Notebook](notebooks/) for detailed documentation and analysis on both synthetic and real datasets.

### Data and Pretrained Model

- Access the pretrained model and data at [Pretrained Model](/models) and [Data](data/) respectively.

## License

Cryo-forum is released as open source software under the [GNU General Public License, Version 3](https://github.com/phonchi/Cryo-forum/blob/master/LICENSE).

## Credits

- This code base builds upon the work developed by [Jelena Banjac](https://github.com/JelenaBanjac/protein-reconstruction) and the code available at [Bingham Rotation Learning Repository](https://github.com/utiasSTARS/bingham-rotation-learning/tree/1c5ee6626f99522f85f1d6c106b1230be9be09cc).
- The visualization code for the Bingham distribution is adapted from [this source](https://github.com/woven-planet/BinghamNLL/tree/main).
