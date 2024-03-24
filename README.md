[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Computational Notebook for "Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter"

<h5>Gregory J. Baker<sup>1,2,3,*</sup>,    
Edward Novikov<sup>1,2,4,</sup>,
Ziyuan Zhao<sup>5</sup>,
Tuulia Vallius<sup>1,2</sup>,
Janae A. Davis<sup>6</sup>,
Jia-Ren Lin<sup>2</sup>,
Jeremy L. Muhlich<sup>2</sup>,
Elizabeth A. Mittendorf<sup>6,7,8</sup>,
Sandro Santagata<sup>1,2,3,9</sup>,
Jennifer L. Guerriero <sup>1,2,6,7,8</sup>,
Peter K. Sorger<sup>1,2,3,*</sup></h5>

<sup>1</sup>Ludwig Center for Cancer Research at Harvard, Harvard Medical School, Boston, MA<br>
<sup>2</sup>Laboratory of Systems Pharmacology, Program in Therapeutic Science, Harvard Medical
School, Boston, MA<br>
<sup>3</sup>Department of Systems Biology, Harvard Medical School, Boston, MA<br>
<sup>4</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA<br>
<sup>5</sup>Systems, Synthetic, and Quantitative Biology Program, Harvard University, Cambridge, MA<br>
<sup>6</sup>Breast Tumor Immunology Laboratory, Dana-Farber Cancer Institute, Boston, MA<br>
<sup>7</sup>Breast Oncology Program, Dana-Farber/Brigham and Women's Cancer Center, Boston, MA<br>
<sup>8</sup>Division of Breast Surgery, Department of Surgery, Brigham and Women's Hospital, Boston, MA<br>
<sup>9</sup>Department of Pathology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA<br>
\*Corresponding Authors: gregory_baker2@hms.harvard.edu (G.J.B.), peter_sorger@hms.harvard.edu (P.K.S)<br>

[BioRxiv preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1)

## Abstract
Tumors are complex assemblies of cellular and acellular structures patterned on spatial scales from microns to centimeters. Study of these assemblies has advanced dramatically with the introduction of high-plex spatial profiling. Image-based profiling methods reveal the intensities and spatial distributions of 20-100 proteins at subcellular resolution in 10<sup>3</sup>–<sup>7</sup> cells per specimen. Despite extensive work on methods for extracting single-cell data from these images, all tissue images contain artefacts such as folds, debris, antibody aggregates, optical aberrations and image processing errors that arise from imperfections in specimen preparation, data acquisition, image assembly, and feature extraction. We show that these artefacts dramatically impact single-cell data analysis, obscuring meaningful biological interpretation. ¬We describe an interactive quality control software tool, CyLinter, that identifies and removes data associated with imaging artefacts. CyLinter greatly improves single-cell analysis, especially for archival specimens sectioned many years prior to data collection, such as those from clinical trials.

## Running the computational notebook
The Python code (i.e., Jupyter Notebooks) in this GitHub repository was used to generate figures in the paper. To run the code, first clone this repo onto your computer. Then download the [datasets](https://www.synapse.org/#!Synapse:syn24193163/files/) from the Sage Bionetworks Synpase data repository into the top level of the cloned repo. Next, change directories into the top level of the cloned repo and create and activate a dedicated Conda environment with the necessary Python libraries by running the following two commands:

```bash
conda create -n cylinter-paper python=3 --file requirements.txt
conda activate cylinter-paper

```

Run the computational notebook in JupyterLab with this command:
```bash
jupyter lab

```

---


## CyLinter Documentation

![](./docs/cylinter-logo.svg)

CyLinter software is written in Python3, archived on the Anaconda package repository, version controlled on [Git/GitHub](https://github.com/labsyspharm/cylinter), instantiated as a configurable Python Class object, and validated for Mac and PC operating systems. Information on how to install and run the program is available at the [CyLinter project website](https://labsyspharm.github.io/cylinter/). 

---

## CyLinter Workflow

![Summary figure](./docs/ExtFig4.png)

**Identifying and removing noisy single-cell data points with CyLinter.** **|** CyLinter input consists of multiplex microscopy files (OME-TIFF/TIFF) and their corresponding cell segmentation outlines (OME-TIFF/TIFF), cell ID masks (OME-TIFF/TIFF), and single-cell feature tables (CSV). **a**, Aggregate data (automated): raw spatial feature tables for all samples in a batch are merged into a single Pandas (Python) dataframe. **b**, ROI selection (interactive or automated): multi-channel images are viewed to identify and gate on regions of tissue affected by microscopy artefacts (negative selection mode) or areas of tissue devoid of artefacts (positive selection mode. **b1-b4**, Demonstration of automated artefact detection in CyLinter: **b1**, CyLinter’s selectROIs module showing artefacts in the CDKN1A (green) channel of a mesothelioma TMA core. **b2**, Transformed version of the original CDKN1A image such that artefacts appear as large, bright regions relative to channel intensity variations associated with true signal of immunoreactive cells which are suppressed. **b3**, Local intensity maxima are identified in the transformed image and a flood fill algorithm is used to create a pixel-level binary mask indicating regions of tissue affected by artefacts. In this example, the method identifies three artefacts in the image: one fluorescence aberration at the top of the core, and two tissue folds at the bottom of the core. **b4**, CyLinter’s selectROIs module showing the binary artefact mask (translucent gray shapes) and their corresponding local maxima (red dots) defining each of the three artefacts. **c**, DNA intensity filter (interactive): histogram sliders are used to define lower and upper bounds on nuclear counterstain single intensity. Cells between cutoffs are visualized as scatter points at their spatial coordinates in the corresponding tissue for gate confirmation or refinement. **d**, Segmentation area filter (interactive): histogram sliders are used to define lower and upper bounds on cell segmentation area (pixel counts). Cells between cutoffs are visualized as scatter points at their spatial coordinates in the corresponding tissue for gate confirmation or refinement. **e**, Cross-cycle correlation filter (interactive): applicable to multi-cycle experiments. Histogram sliders are used to define lower and upper bounds on the log-transformed ratio of DNA signals between the first and last imaging cycles. Cells between cutoffs are visualized as scatter points at their spatial coordinates in their corresponding tissues for gate confirmation or refinement. **f**, Log transformation (automated): single-cell data are log-transformed. **g**, Channel outliers filter (interactive): the distribution of cells according to antibody signal intensity is viewed for all sample as a facet grid of scatter plots (or hexbin plots) against cell area (y-axes). Lower and upper percentile cutoffs are applied to remove outliers. Outliers are visualized as scatter points at their spatial coordinates in their corresponding tissues for gate confirmation or refinement. **h**, MetaQC (interactive): unsupervised clustering methods (UMAP or TSNE followed by HDBSCAN clustering) are used to correct for gating bias in prior data filtration modules by thresholding on the percent of each cluster composed of clean (maintained) or noisy (redacted) cells. **i**, Principal component analysis (PCA, automated): PCA is performed and Horn’s parallel analysis is used to determine the number of PCs associated with non-random variation in the dataset. **j**, Image contrast adjustment (interactive): channel contrast settings are optimized for visualization on reference tissues which are applied to all samples in the cohort. **k**, Unsupervised clustering (interactive): UMAP (or TSNE) and HDBSCAN are used to identify unique cell states in a given cohort of tissues. Manual gating can also be performed to identify cell populations. **l**, Compute clustered heatmap (automated): clustered heatmap is generated showing channel z-scores for identified clusters (or gated populations). **m**, Compute frequency statistics (automated): pairwise t statistics on the frequency of each identified cluster or gated cell population between groups of tissues specified in CyLinter’s configuration file (cylinter_config.yml, e.g., treated vs. untreated, response vs. no response, etc.) are computed. **n**, Evaluate cluster membership (automated): cluster quality is checked by visualizing galleries of example cells drawn at random from each cluster identified in the clustering module (panel k). Visit the CyLinter project website at https://labsyspharm.github.io/cylinter/) for further details.</h6>

---

## Data Availability

See **Supplementary Table 1** of the CyLinter manuscript for a complete list of datasets and their associated identifiers and repositories.

---


## Image Processing

The whole-slide and tissue microarray images analyzed in this study were preprocessed using the [MCMICRO](https://mcmicro.org/) [[2]](#2) image assembly and feature extraction pipeline.

---


## Funding

This work was supported by the Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.) and by NIH NCI grants U2C-CA233280, and U2C-CA233262 (P.K.S., S.S.). Development of computational methods and image processing software is supported by a Team Science Grant from the Gray Foundation (P.K.S., S.S.), the Gates Foundation grant INV-027106 (P.K.S.), the David Liposarcoma Research Initiative at Dana-Farber Cancer Institute supported by KBF Canada via the Rossy Foundation Fund (P.K.S., S.S.) and the Emerson Collective (P.K.S.). S.S. is supported by the BWH President’s Scholars Award.

---

## Zenodo Archive

The Python code (i.e., Jupyter Notebooks) in this GitHub repository is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)

---


## References

<a id="1">[1]</a>
Baker GJ. et al. Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter. **bioRxiv** (2023) https://doi.org/10.1101/2023.11.01.565120

<a id="1">[2]</a>
Schapiro D., Sokolov A., Yapp C. et al. MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging. **Nature Methods** 19, 311–315 (2022). https://doi.org/10.1038/s41592-021-01308-y


