# Healthy vs diseased tooth and implant plaque biofilms

Identify characteristic spatial structures and/or determine the range of spatial structures in plaque biofilms.

HSDM plaque biofilm hiprfish image processing

Mix of tooth and implant, healthy and diseased

4 different datasets: 
        2023_02_18
        2023_02_08
        2023_10_16
        2022_12_16

Mix of encodings, commonly with 5bit and 3 lasers (488, 514, 561), one set with 7bit and 4 lasers (add 633).

Commonly tiled images or multiple scenes. 

### nb_evaluate_images.ipynb 

I've written a script to register stage shifts between lasers and segment each tile individually and output segmented .npy file and cell properties including average spectrum in .czv file as well as an rgb plot of the raw signal and a segmentation plot (I used functions from https://github.com/benjamingrodner/hipr_mge_fish). 

I added a section to correct shifts calculated using the phase cross correlation that were outliers from the other shifts in the other tiles. I replaced deviating shifts with the median shift for that laser. 

After segmentation there are 984,005 cells

### nb_stitching.ipynb

I tried to figure out [m2stitch](https://github.com/yfukai/m2stitch/tree/master), but it was not working. 

### Snakemake pipeline

I ran segmentation in nb_evaluate_images.ipynb, but transferred the code over to Snakefile...still need to test. 

I ran the images through the pipeline for cluster, classif, and get_coords rules. 

input_table.csv needs full filepaths containing the date at the beginning as yyyy_mm_dd and "fov_\d\d" right before the laser identifier. Include all laser czis in the table, they get grouped together. 

Lots of config stuff still in the snakemake like defining colors for taxa, specifying barcode type, and path to barcode assignment file. 

Input and output stuff is in the config.yaml file. 

Usage: 

```
snakemake --configfile config.yaml -j 20 -p
```

