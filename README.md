# Harvard-GF

The code and dataset for the paper entitled [**Harvard Glaucoma Fairness: A Retinal Nerve Disease Dataset for Fairness Learning and Fair Identity Normalization**](https://ieeexplore.ieee.org/abstract/document/10472539). The abbreviation of our dataset is **Harvard-GF**, which stands for Harvard Glaucoma Fairness. Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

# Dataset

The dataset can be accessed via this [link](https://drive.google.com/drive/folders/1-38HdWTqR4RH5GYT4bBtYu5ADUTN98Gk?usp=drive_link). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.

The glaucoma data of 3,300 patients includes 3,300 OCT RNFLT maps (dimensions 200 x 200 x 200). The visual field, patient age, sex, race, and glaucoma label information are also included in the data. 2100 samples are for training, 300 samples for validation, and 900 samples for testing. A unique strength of this dataset is that we have equal numbers for the three racial groups Asian, Black, and White.

Each data contains the following attributes:
1) rnflt: OCT RNFLT map of size 200 x 200.
2) oct_bscans: 3D OCT B-scans image of size 200 x 200 x 200.
3) glaucoma: glaucomatous status, 0 for non-glaucoma and 1 for glaucoma.
4) md: mean deviation value of visual field.
5) tds: 52 total deviation values of visual field.
6) age: patient age.
7) male: patient sex, 0 for female and 1 for male.
8) race: patient race, 0 for Asian, 1 for Black or African American, and 2 for White or Caucasian.
9) marital status: 0 for Married/Civil Union, Life Partner, 1: Single, 2: Divorced, 3: Widowed, 4: Legally Separated, and -1: Unknown
10) ethnicity: 0: Non-Hispanic, 1: Hispanic, and -1: Unknown
11) language: 0: English, 1: Spanish, 2: Others, and -1: Unknown


# Abstract

Fairness (also known as equity interchangeably) in machine learning is important for societal well-being, but limited public datasets hinder its progress. Currently, no dedicated public medical datasets with imaging data for fairness learning are available, though minority groups suffer from more health issues. To address this gap, we introduce Harvard Glaucoma Fairness (Harvard-GF), a retinal nerve disease dataset with both 2D and 3D imaging data and balanced racial groups for glaucoma detection. Glaucoma is the leading cause of irreversible blindness globally with Blacks having doubled glaucoma prevalence than other races. We also propose a fair identity normalization (FIN) approach to equalize the feature importance between different identity groups. Our FIN approach is compared with various the-state-of-the-art fairness learning methods with superior performance in both racial and gender fairness tasks with 2D and 3D imaging data, which demonstrate the utilities of our dataset Harvard-GF for fairness learning. To facilitate fairness comparisons between different models, we propose an equity-scaled performance measure, which can be flexibly used to compare all kinds of performance metrics in the context of fairness.

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
## Acknowledgement & Citation


If you find this repository useful for your research, please consider citing our [paper](https://ieeexplore.ieee.org/abstract/document/10472539):

```bibtex
@article{10472539,
  author={Luo, Yan and Tian, Yu and Shi, Min and Pasquale, Louis R. and Shen, Lucy Q. and Zebardast, Nazlee and Elze, Tobias and Wang, Mengyu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Harvard Glaucoma Fairness: A Retinal Nerve Disease Dataset for Fairness Learning and Fair Identity Normalization}, 
  year={2024},
  volume={43},
  number={7},
  pages={2623-2633},
  keywords={Glaucoma;Biomedical imaging;Data models;Finance;Three-dimensional displays;Medical services;Measurement;AI for eye disease screening;equitable deep learning;fairness learning},
  doi={10.1109/TMI.2024.3377552}}
