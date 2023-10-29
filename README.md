[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8371088.svg)](https://doi.org/10.5281/zenodo.8371088)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter

Gregory J. Baker<sup>1,2,3,</sup>\*,    
Edward Novikov<sup>1,4,</sup>,
Ziyuan Zhao<sup>5</sup>,
Tuulia Vallius<sup>1,2</sup>,
Janae A. Davis<sup>6</sup>,
Jia-Ren Lin<sup>2</sup>,
Jeremy L. Muhlich<sup>2</sup>,
Elizabeth A. Mittendorf<sup>6,7,8</sup>,
Sandro Santagata<sup>1,2,3,9</sup>,
Jennifer L. Guerriero <sup>1,2,6,7,8</sup>,
Peter K. Sorger<sup>1,2,3</sup>\*

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

---


## Scientific summary

Tumors are complex assemblies of cellular and acellular structures patterned on spatial scales from microns to centimeters. Study of these assemblies has advanced dramatically with the introduction of methods for highly multiplexed tissue imaging methods. These reveal the intensities and spatial distributions of 20-100 proteins in 10<sup>3</sup>–<sup>7</sup> cells per specimen in a preserved tissue microenvironment. Despite extensive work on extracting single-cell image data, all tissue images are afflicted by artifacts (e.g., lint, antibody aggregates) that arise from unreliable staining of features such as necrotic domains and imperfections in specimen preparation and data acquisition. We show that artifacts dramatically impact single-cell data analysis, in extreme cases, preventing meaningful biological interpretation. We describe an interactive quality control software tool, CyLinter, that identifies and removes data associated with imaging artifacts. CyLinter greatly improves single-cell analysis, especially for archival specimens sectioned many years prior to data collection, including those from clinical trials.

![Summary figure](./docs/ExtFig5.png)

---


## CyLinter documentation

![](./docs/cylinter-logo.svg)

CyLinter software is written in Python3, archived on the Anaconda package repository, versioned controlled on [Git/GitHub](https://github.com/labsyspharm/cylinter), instantiated as a configurable Python Class object, and validated for Mac and PC operating systems. Information on how to install and run the program is available at the [CyLinter website](https://labsyspharm.github.io/cylinter/). 

---


## Data availability

New data associated with this paper is available at the [HTAN Data Portal](https://data.humantumoratlas.org). Previously published data is through public repositories. See Supplementary Table 1 for a complete list of datasets and their associated identifiers and repositories. Online Supplementary Figures 1-4 and the CyLinter demonstration dataset can be accessed at [Sage Synapse](https://www.synapse.org/#!Synapse:syn24193163/files).

---


## Image processing

The whole-slide and tissue microarray images described in this study were processed using [MCMICRO](https://mcmicro.org/) [[1]](#1) image assembly and feature extraction pipeline.

---


## Funding and Acknowledgments

This work was supported by the Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.) and by NIH NCI grants U54-CA225088, U2C-CA233280, and U2C-CA233262 (P.K.S., S.S.). Development of computational methods and image processing software is supported by a Team Science Grant from the Gray Foundation (P.K.S., S.S.), the Gates Foundation grant INV-027106 (P.K.S.), the David Liposarcoma Research Initiative (P.K.S., S.S.), Emerson Collective (P.K.S.). S.S. is supported by the BWH President’s Scholars Award. We gratefully acknowledge Juliann Tefft for superb editorial support; Kai Wucherpfennig and Sascha Marx for providing the HNSCC CODEX dataset; Zoltan Maliga and Connor Jacobson for providing CyCIF EMIT TMA 22 images; and the Dana-Farber/Harvard Cancer Center for use of the Specialized Histopathology Core, which provided TMA construction and sectioning services. We also thank Yu-An Chen for assisting in the collection of CyCIF data from the SARDANA-097 tissue sample performed as part of the National Cancer Institute (NCI) Human Tumor Atlas Network (HTAN). 

---

## References

<a id="1">[1]</a>
Schapiro, D., Sokolov, A., Yapp, C. et al. MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging. **Nature Methods** 19, 311–315 (2022). https://doi.org/10.1038/s41592-021-01308-y

