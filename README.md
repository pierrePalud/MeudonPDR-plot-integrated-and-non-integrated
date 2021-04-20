# MeudonPDR-plot-integrated-and-non-integrated

For any line, plots maps of a few integrated parameters (intensity, column density) and non integrated parameters (temperature, volume density, optical depth) at an approximation of the emissivity peak.

# Installation

To install : 

```
git clone <>
cd MeudonPDR-plot-integrated-and-non-integrated
conda env create -f environment.yml
```


# How to use

To use, first add a `./data` folder with your data, then run : 

```
conda activate integrated-non-integrated-env
python utils.py
```

which should give you the following plot :

![Alt text](./img/example.png?raw=true "Example of plot this program does")
