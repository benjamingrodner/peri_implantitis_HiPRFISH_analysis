# Microscale Spatial Dysbiosis in Oral biofilms Associated with Disease

This repository contains the specific code used to generate the figures in "Microscale Spatial Dysbiosis in Oral biofilms Associated with Disease"

## Acknowledgement

This code makes use of open source packages including `numpy`, `pandas`, `aicsimageio`, `scikit-image`, `scikit-learn`, `PySAL`, `OpenCV`, `scipy`, `turbustat`, and `matplotlib`.

This code also makes use of [code](https://github.com/proudquartz/hiprfish) developed in [Shi, et al. 2020](https://doi.org/10.1038/s41586-020-2983-4). 

## Image processing

Create a conda environment using:

```
conda env create -f env_image_processing.yml
conda activate env_image_processing
```

Cythonize "functions/neighbor2d.pyx" as in this [tutorial](https://docs.cython.org/en/latest/src/quickstart/build.html). 

List input image filenames in the format of "input_table_test.csv"

Adjust config.yaml to point to the input table, functions, genus barcode, and output directories. You can also adjust image processing parameters here. 

Run the image processing pipeline

```
snakemake -s Snakefile_image_processing --configfile config.yaml -j NUM_CORES -p
```
## Cell density power spectrum

Create a new conda environment:

```
conda env create --f env_turbustat.yml
conda activate env_turbustat
```

Run the power spectrum pipeline

```
snakemake -s Snakefile_turbustat --configfile config.yaml -j NUM_CORES -p
```

For a more hands on look at the power spectrum analysis see "nb_turbustat.ipynb"

## Spatial analysis

Create a new conda environment:

```
conda env create --f env_spatial.yml
conda activate env_spatial
```

Run the power spectrum pipeline

```
snakemake -s Snakefile_spatial --configfile config.yaml -j NUM_CORES -p
```

Some of the plotting is done manually in "nb_spatial_stats.ipynb"