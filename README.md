# SLOTS
Semi-Supervised End-to-End  Contrastive Learning for Time Series Classification

## Overview

This repository contains two processed datasets (except for three other datasets that are too large) and the codes of SLOTS (along with baselines) for manuscript Semi-Supervised End-To-End Contrastive Learning For Time Series Classification. We propose an end-to-end model called SLOTS (Semi-supervised Learning fOr Time clasSification). We evaluate SLOTS by comparing it with ten state-of-the-art methods across five datasets.The following illustration provides an overview of the conventional two-stage framework and our end-to-end framework. 
![image](https://github.com/DL4mHealth/SLOTS/assets/47804803/33b8eb33-7691-473b-8884-29bcc63ae157)

## Key idea of SLOTS

In this paper, we present a novel semi-supervised framework that achieves optimal performance with minimal labeled samples and can be seamlessly integrated into various architectures. Our approach systematically combines unsupervised contrastive loss, supervised contrastive loss, and classification loss to jointly update the model, maximizing the utilization of information from the data. We present the model pipeline of the proposed SLOTS in Figure 2.
![image](https://github.com/DL4mHealth/SLOTS/assets/47804803/66cef79e-49fb-455f-a9d2-54b27f14ed48)

## Datasets

We evaluate the SLOTS model on five time series datasets covering a large set of variations: different numbers of subjects (from 15 to 38,803), different scenarios (neurological healthcare, human activity recognition, physical status monitoring, etc.), and diverse types of signals (EEG, acceleration, and vibration, etc.).

(1) [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html), a multimodal dataset based on the stimulation of music video materials, is used to analyze human emotional states. It contains 32 subjects monitored by 32-channel EEG signals while watching 40 minutes (each minute is a trail) of videos.

(2) [SEED](https://bcmi.sjtu.edu.cn/home/seed/) includes 62-channel EEG data of 15 subjects when they are watching 15 film clips with three types of emotions.

(3) [EPILEPSY](https://repositori.upf.edu/handle/10230/42894) monitors the brain activities of 500 subjects with a single-channel EEG sensor (174Hz). Each sample is labeled in binary based on whether the subject has epilepsy or not. 

(4) [HAR](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) has 10,299 9-dimension samples from 6 daily activities. 

(5) [P19](https://physionet.org/content/challenge-2019/1.0.0/) (PhysioNet Sepsis Early Prediction Challenge 2019) includes 38,803 patients that are monitored by 34 sensors in ICU.

## Requirements

SLOTS has been tested using Python >=3.9.


## License

SLOTS is licensed under the MIT License.
