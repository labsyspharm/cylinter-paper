[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Computational Notebook for "Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter"

<h5>Gregory J. Baker<sup>1,2,3,*</sup>,    
Edward Novikov<sup>1,4,</sup>,
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

<!-- *Nature Cancer (2023). DOI: [10.1038/s43018-023-00576-1](https://doi.org/10.1038/s43018-023-00576-1)* -->

## Abstract

Tumors are complex assemblies of cellular and acellular structures patterned on spatial scales from microns to centimeters. Study of these assemblies has advanced dramatically with the introduction of methods for highly multiplexed tissue imaging. These reveal the intensities and spatial distributions of 20-100 proteins in 10<sup>3</sup>–<sup>7</sup> cells per specimen in a preserved tissue microenvironment. Despite extensive work on extracting single-cell image data, all tissue images are afflicted by artifacts (e.g., folds, debris, antibody aggregates, optical effects, image processing errors) that arise from imperfections in specimen preparation, data acquisition, image assembly, and feature extraction. We show that artifacts dramatically impact single-cell data analysis, in extreme cases, preventing meaningful biological interpretation. ¬We describe an interactive quality control software tool, CyLinter, that identifies and removes data associated with imaging artifacts. CyLinter greatly improves single-cell analysis, especially for archival specimens sectioned many years prior to data collection, including those from clinical trials.

The Python code (i.e., Jupyter Notebooks) in this GitHub repository was used to generate the figures in the aforementioned study.

[Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1)

---



## CyLinter Workflow

![Summary figure](./docs/ExtFig5.png)

**Identifying and Removing Noisy Single-cell Data Points with CyLinter.** **|** **a-d**: CyLinter input: **a**, Multiplex microscopy file, **b**, Cell segmentation outlines, **c**, Cell ID mask, **d**, Single-cell feature table. **e**, ROI selection module: multi-channel images are viewed to identify and gate on regions of tissue affected by microscopy artifacts (in the default negative selection mode). **f-i**, Demonstration of automated artifact detection in CyLinter. **f**, CyLinter’s selectROIs module showing artifacts in the CDKN1A (green) channel of EMIT TMA core 18 (mesothelioma). **g**, Transformed version of the original CDKN1A image such that artifacts appear as large, bright regions relative to channel intensity variations associated with true signal of immunoreactive cells which are suppressed. **h**, Local intensity maxima are identified in the transformed image and a flood fill algorithm is used to create a pixel-level binary mask indicating regions of tissue affected by artifacts. In this example, the method identifies three artifacts in the image: one fluorescence aberration at the top of the core, and two tissue folds at the bottom of the core. **i**, CyLinter’s selectROIs module showing the binary artifact mask (translucent gray shapes) and their corresponding local maxima (red dots) defining each of the three artifacts. **j**, DNA intensity filter: histogram sliders are used to define lower and upper bounds on nuclear counterstain single intensity. Cells between cutoffs are visualized as scatter points at their spatial coordinates in the corresponding tissue for gate confirmation or refinement. **k**, Cell segmentation area filter: histogram sliders are used to define lower and upper bounds on cell segmentation area (pixel counts). Cells between cutoffs are visualized as scatter points at their spatial coordinates in the corresponding tissue for gate confirmation or refinement. **l**, Cross-cycle correlation filter: applicable to multi-cycle experiments. Histogram sliders are used to define lower and upper bounds on the log-transformed ratio of DNA signals between the first and last imaging cycles. Cells between cutoffs are visualized as scatter points at their spatial coordinates in their corresponding tissues for gate confirmation or refinement. **m**, Channel outlier filter: the distribution of cells according to antibody signal intensity is viewed for all sample as a facet grid of scatter plots (or hexbin plots) against cell area (y-axes). Lower and upper percentile cutoffs are applied to remove outliers. Outliers are visualized as scatter points at their spatial coordinates in their corresponding tissues for gate confirmation or refinement. **n**, MetaQC module: unsupervised clustering methods (UMAP or TSNE followed by HDBSCAN clustering) are used to correct for gating bias in prior data filtration modules by thresholding on the percent of each cluster composed of clean (maintained) or noisy (redacted) cells. **o**, Unsupervised cluster methods (UMAP or TSNE followed by HDBSCAN) are used to identify unique cell states in a given cohort of tissues. **p**, Image contrast adjustment: channel contrast settings are optimized for visualization on reference tissue which are applied to all tissues in the cohort. **q**, Evaluate cluster membership: cluster quality is checked by visualizing galleries of example cells drawn at random from each cluster identified in the clustering module.</h6>

---


## CyLinter Documentation

![](./docs/cylinter-logo.svg)

CyLinter software is written in Python3, archived on the Anaconda package repository, versioned controlled on [Git/GitHub](https://github.com/labsyspharm/cylinter), instantiated as a configurable Python Class object, and validated for Mac and PC operating systems. Information on how to install and run the program is available at the [CyLinter website](https://labsyspharm.github.io/cylinter/). 

---


## Data Availability

New data associated with this paper is available at the [HTAN Data Portal](https://data.humantumoratlas.org). Previously published data is through public repositories. See Supplementary Table 1 for a complete list of datasets and their associated identifiers and repositories. Online Supplementary Figures 1-4 and the CyLinter demonstration dataset can be accessed at [Sage Synapse](https://www.synapse.org/#!Synapse:syn24193163/files)


---


## Image Processing

The whole-slide and tissue microarray images described in this study were processed using [MCMICRO](https://mcmicro.org/) [[2]](#2) image assembly and feature extraction pipeline.

---


## Funding and Acknowledgments

This work was supported by the Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.) and by NIH NCI grants U2C-CA233280, and U2C-CA233262 (P.K.S., S.S.). Development of computational methods and image processing software is supported by a Team Science Grant from the Gray Foundation (P.K.S., S.S.), the Gates Foundation grant INV-027106 (P.K.S.), the David Liposarcoma Research Initiative (P.K.S., S.S.) and the Emerson Collective (P.K.S.). S.S. is supported by the BWH President’s Scholars Award. We gratefully acknowledge Juliann Tefft for superb editorial support; Kai Wucherpfennig and Sascha Marx for providing the HNSCC CODEX dataset; Zoltan Maliga and Connor Jacobson for providing CyCIF EMIT TMA22 images; and the Dana-Farber/Harvard Cancer Center for use of the Specialized Histopathology Core, which provided TMA construction and sectioning services. We also thank Yu-An Chen for assisting in the collection of CyCIF data from the SARDANA-097 tissue sample performed as part of the National Cancer Institute (NCI) Human Tumor Atlas Network (HTAN).

---

## Zenodo Archive

The Python code (i.e., Jupyter Notebooks) in this GitHub repository is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10067804.svg)](https://doi.org/10.5281/zenodo.10067804)

---


## References

<a id="1">[1]</a>
Baker GJ. et al. Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter. **bioRxiv** (2023) https://doi.org/10.1101/2023.11.01.565120

<a id="1">[2]</a>
Schapiro D., Sokolov A., Yapp C. et al. MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging. **Nature Methods** 19, 311–315 (2022). https://doi.org/10.1038/s41592-021-01308-y


