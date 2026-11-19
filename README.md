# Adult initialization and region-specific quality assessment for pediatric high-grade glioma segmentation with nnU-Net

Study materials for Wang, Chen, Wu, Wu, Huang, Feng and Wang, *Frontiers in Neurology*, article 2009806. [Article DOI](https://doi.org/10.3389/fneur.2026.2009806)

This study compares pediatric-only training with initialization from one adult-pretrained checkpoint. The pediatric cohort contains 257 annotated MRI examinations from CBTN, DFCI and DIPGr. Each source was held out in turn for model fitting, and pediatric training used three seeds (2026, 2027 and 2028). Earlier development included access to DIPGr. The historical geometric planning used the full cohort.

## Using the release

| Directory | What to use it for |
| --- | --- |
| [cohort](cohort/) | Match the released case identifiers to the source datasets and construct the recorded training and held-out splits. |
| [replication](replication/) | Prepare data, configure the training environment and run the model workflows described in the accompanying instructions. |
| [analysis](analysis/) | Recalculate results from the released evaluation records and reproduce the documented figures. |
| [publication_figures](publication_figures/) | Regenerate the final axis-labelled vector versions of Figures 2, 3, 4, 5 and 7. |

This is the code and analysis release. The training scripts use nnU-Net as an external dependency. Trained model weights and raw MRI data are outside this archive.

To reproduce the main numerical results from the included records, run the following from this directory with an analysis Python environment:

```sh
python -m pip install -r analysis/requirements.txt
python analysis/run.py
```

The paired analysis of the 171 enhancing-tumor-positive examinations is a useful first check. The across-seed case mean Dice is 0.5897 for pediatric-only training and 0.6086 for adult initialization. The paired difference is 0.0189 (95% case-bootstrap interval 0.0054 to 0.0322). In the two additional seeds the paired difference is 0.0172 (0.0007 to 0.0327). The comparison of planning-related shared omissions uses 136 enhancing-tumor-positive examinations from the two sources whose plans changed.

## Source data

Obtain the MRI and expert reference masks from the source repositories under their respective access and reuse conditions. Case identifiers in this release refer to the public collections. The pediatric data are from the BraTS-PEDs collection, Fathi Kazerooni et al., The Cancer Imaging Archive. Adult pretraining used the BraTS 2021 training collection, Baid et al., The Cancer Imaging Archive. [BraTS-PEDs dataset](https://doi.org/10.7937/DX5C-TJ86) · [BraTS 2021 dataset](https://doi.org/10.7937/JC8X-9874) · [BraTS 2021 challenge data instructions](https://www.med.upenn.edu/cbica/brats2021/)

## Integrity

`SHA256SUMS.txt` lists the SHA-256 digest of each release file, excluding the checksum list itself. Run `shasum -a 256 -c SHA256SUMS.txt` from the extracted directory on macOS, or `sha256sum -c SHA256SUMS.txt` on Linux.
