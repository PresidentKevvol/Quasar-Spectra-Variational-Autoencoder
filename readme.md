# README for Latent Space Analysis of Quasar Spectra Features using the Variational Autoencoder InfoVAE

By Erik Weiss and Kevin Hui, a project for EECS 6412 Data Mining.

There will be 5 files of main interest, listed as follows:
- dataset_poke_01.ipynb
- dataset_poke_02.ipynb
- InfoVAE_2.py
- variational_autoencoder_04a1.ipynb
- variational_autoencoder_04a2.ipynb
- variational_autoencoder_04x_1.ipynb
- examine_VAEs_02.ipynb

And the important data files:
- dr16qsnr10_dataset_vae_eval.npy
- dr16qsnr10_errterm_vae_eval.npy
- v4fits_important_values_hmap.pkl

Kindly download the larger files (`dr16qsnr10_dataset_vae_eval.npy` and `dr16qsnr10_errterm_vae_eval.npy`) [from here](https://drive.google.com/drive/folders/1qZ4yPkS04tjRJPzOnn9hjHANgcrTtfPm?usp=sharing)
and put them in the directory of this repo after you pull/download it.

And the stored model files:
- VAE_3feat_lambd-5.043311_alpha-0.000000_nhidden-256-128.pth
- VAE_6feat_lambd-6.059668_alpha-0.000000_nhidden-256-128.pth
- VAE_9feat_lambd-8.547600_alpha-0.000000_nhidden-256-128.pth
- VAE_12feat_lambd-8.389687_alpha-0.000000_nhidden-256-128.pth
- VAE_15feat_lambd-7.823182_alpha-0.000000_nhidden-256-128.pth

Many of the code here are borrowed from the repository by [Portillo et al. 2020](https://github.com/stephenportillo/SDSS-VAE)

# How to use the code files

## Preparing The Dataset

### Code to execute:
Nothing here really (raw dataset too large to put in this repository)

### Details
`dataset_poke_01.ipynb` and `dataset_poke_02.ipynb` performs interpolation on the raw dataset, where spectra with different frequency ranges are unified to a grid with same scale.
The raw data is 21859 files each with about ~4500 'rows', each row represents a 'pixel' that has:
- a wavelength,
- the flux (amount of light energy accepted) of that wavelength,
- error term denoting how much error this pixel have,
- and a few other columns not used
but the frequency of the spectra files are not aligned, thus pre-processing is needed.
The new grid is a wavelength range from 3625 - 9000 Angstrom, 3000 elements spaced evenly in log-linear scale.
The flux values are interpolated using a 4th degree polynomial interpolation
The pixel error values are interpolated using max value between closest two
A few data files that does not cover the full range is discarded, thus, subsequent stages works with 21771 spectra.

### Notice

Due to storage constraints we are not able to provide the full raw dataset here. However subsequent steps can still be ran to reproduce results using the intermediate dataset files.
The interpolated data is further subsampled to 1500 element arrays for each quasar, stored in `dr16qsnr10_dataset_vae_eval.npy` and `dr16qsnr10_errterm_vae_eval.npy` for convinient access.
The above two are `.npy` array files of a sized (21771, 1500) numpy array, and can be loaded by using `np.load()`.
`v4fits_important_values_hmap.pkl` is a python pickle file of a dict that stores useful values extracted from another dataset file called [DR16Q_v4](https://data.sdss.org/datamodel/files/BOSS_QSO/DR16Q/DR16Q_v4.html).
The values in it are properties that describes the quasars' astrophysical attributes and are used for subsequent qualitative analyses.


## Training the VAEs with hyperparameter optimization

### Code to execute
- variational_autoencoder_04a1.ipynb
- variational_autoencoder_04a2.ipynb
- variational_autoencoder_04x_1.ipynb

### Details
`variational_autoencoder_04a1.ipynb` and `variational_autoencoder_04a2.ipynb` are the notebook files that trains the VAEs.
It imports code from `InfoVAE_2.py`.
It is chopped into two parts due to an error we made and time constraint with fixing the error.

It creates multiple VAE models using the PyTorch framework and perform a hyperparameter search using SciKitLearn Optimization (skopt) to look for the best lambda value - a parameter used in the loss function for training the VAEs that is a weight parameter of a weighted average combining a few information theoretic metrics.
It tried 5 different configurations - VAE with {3, 6, 9, 12, 15} latent dimensions.

`variational_autoencoder_04x_1.ipynb` re-creates the VAE with 9 latent dimensions with the best lambda and train it, in order to generate the train/validation loss and MSE graph.

### Notice
The models we created are stored in a directory in this repo called `VAE_dr16qsnr10_hypersearch_2024-12-16d`.
If you do decide to reproduce this section (using `examine_VAEs_02.ipynb`), the optimal lambda values you get might be different.
To inspect them for next part, there is a block with code
```
models_dir = 'VAE_dr16qsnr10_hypersearch_2024-12-16d'
optimal_model_files = [
    'VAE_3feat_lambd-5.043311_alpha-0.000000_nhidden-256-128.pth',
    'VAE_6feat_lambd-6.059668_alpha-0.000000_nhidden-256-128.pth',
    'VAE_9feat_lambd-8.547600_alpha-0.000000_nhidden-256-128.pth',
    'VAE_12feat_lambd-8.389687_alpha-0.000000_nhidden-256-128.pth',
    'VAE_15feat_lambd-7.823182_alpha-0.000000_nhidden-256-128.pth',
]
```
Kindly change the stored models file names to your usage.

## Inspecting the results

### Code to execute
- examine_VAEs_02.ipynb

### Details
`examine_VAEs_02.ipynb` is the notebook that generates the resulting graphs you can see in the paper.

It first start with a plot relating latent dimension count to overall reconstruction MSE (Mean Squared Error of reconstructing each spectra using this model)

It takes the following columns from `v4fits_important_values_hmap.pkl` (DR16Q_v4):
| index | code | data type | description |
| ----- | ---- | --------- | ----------- |
| [1]   | RA   | DOUBLE | Right ascension in decimal degrees (J2000) |
| [2]   | DEC  | DOUBLE | Declination in decimal degrees (J2000) |
| [4]   | MJD  | INT32  | Modified Julian day of the spectroscopic observation |
| [26]  | Z    | DOUBLE | Best available redshift taken from Z_VI, Z_PIPE, Z_DR12Q, Z_DR7Q_SCH, Z_DR6Q_HW, and Z_10K |
| [54]  | NHI_DLA |  DOUBLE[5] | Absorber column density for damped Lyα features |
| [55]  | CONF_DLA | DOUBLE[5] | Confidence of detection for damped Lyα features |
| [56]  | BAL_PROB | FLOAT | BAL probability |
| [57]  | BI_CIV | DOUBLE | BALnicity index for C IV λ1549 region |
| [61]  | BI_SIIV | DOUBLE | BALnicity index for Si IV λ1396 region |
| [90]  | PLATESN2 | DOUBLE | Overall (S/N)2 measure for plate, minimum of all 4 cameras |
| [96]  | M_I | DOUBLE | Absolute i-band magnitude, H0 = 67.6 km s-1 Mpc-1, ΩM = 0.31, ΩL = 0.69, ΩR = 9.11x10-5. K-corrections taken from Table 4 of Richards et al. (2006). Z_PCA used for redshifts |
| [97]  | SN_MEDIAN_ALL | DOUBLE | Median S/N value of all good spectroscopic pixels |
| [158] | XMM_SOFT_FLUX | FLOAT  | Soft (0.2-2.0 keV) X-ray flux from XMM-Newton in erg s-1 cm-2 |
| [160] | XMM_HARD_FLUX | FLOAT  | Hard (2.0-12.0 keV) X-ray flux from XMM-Newton in erg s-1 cm-2 |
| [162] | XMM_TOTAL_FLUX | FLOAT | Total (0.2-12.0 keV) X-ray flux from XMM-Newton in erg s-1 cm-2 |

And generate 3 plots:
- the histogram denoting the numerical distribution of the column
- the density plot relating the column to Reconstruction MSE (of each individual spectrum)
- Corner plot relating every pair of VAE latent dimensions color coded with the column's value

Then, a plot of 6 sample spectra and their reconstruction are created.

Finally, a corner plot relating the columns Redshift, MJD, absolute i-band magnitude, BAL probability, and median S/N values are created.