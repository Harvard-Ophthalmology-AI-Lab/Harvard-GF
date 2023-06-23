# Harvard-GF

The dataset and code for the paper entitled *Harvard Glaucoma Fairness (Harvard-GF): A Retinal Nerve Disease Dataset for Fairness Learning and Fair Identity Normalization*.

# Dataset

The RNFLT maps with labels and social identies can be accessed via this [link](https://doi.org/10.7910/DVN/A4XMO1).

The glaucoma data of 3300 patients includes 3300 OCT RNFLT maps (dimension 200x200x200). The visual field, patient age, sex, race, glaucoma label information are also included in the data. 2100 samples are for training, 300 samples for validation, and 900 samples for testing.

Each data contains the following attributes:
1) rnflt: OCT RNFLT map of size 200x200.
2) oct_bscans: 3D OCT B-scans image of size 200x200x200.
3) glaucoma: glaucomatous status, 0 for non-glaucoma and 1 for glaucoma.
4) md: mean deviation value of visual field.
5) tds: 52 total deviation values of visual field.
6) age: patient age.
7) male: patient sex, 0 for female and 1 for male.
8) race: patient race, 0 for Asian, 1 for Black or African American, and 2 for White or Caucasian.
9) marital status: 0 for Married/Civil Union, Life Partner, 1: Single, 2: Divorced, 3: Widowed, 4: Legally Separated, and -1: Unknown
10) ethnicity: 0: Non-Hispanic, 1: Hispanic, and -1: Unknown
11) languge: 0: English, 1: Spanish, 2: Others, and -1: Unknown


# Abstract

Fairness in machine learning is important for societal well-being, but limited public datasets hinder its progress. Currently, no dedicated public medical datasets with imaging data for fairness learning are available, though minority groups suffer from more health issues. To address this gap, we introduce EyeFair, a retinal nerve disease dataset with both 2D and 3D imaging data and balanced racial groups for glaucoma detection. Glaucoma is the leading cause of irreversible blindness globally with Blacks having doubled glaucoma prevalence than other races. We also propose a fair-identity normalization (FIN) approach to equalize the feature importance between different identity groups. Our FIN approach is compared with various the-state-of-the-arts fairness learning methods with superior performance in both racial and gender fairness tasks with 2D and 3D imaging data, which demonstrate the utilities of our dataset EyeFair for fairness learning. To facilitate fairness comparisons between different models, we propose an equity-scaled performance measure, which can be flexibly used to compare all kinds of performance metrics in the context of fairness.

# Requirements

The project is based on PyTorch 1.13.1+. To install, execute

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

# Experiments

To run the experiments, execute the scripts in [scripts](./scripts), e.g.,

```
./scripts/train_glaucoma_fair_fin.sh
```
