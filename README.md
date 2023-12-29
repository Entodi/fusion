# Fusion
Fusion is a self-supervised framework for data with multiple sources — specifically, this framework aims to support neuroimaging applications.

Fusion aims to provide a foundation for fair comparison of new models in different multi-view, multi-domain, or multi-modal scenarios. Currently, we provide only two datasets with multi-view, multi-domain natural images. Further, the repository will be updated with multi-modal neuroimaging datasets.

This project is under active development, and the codebase is subject to change.

---
## Installation
To install requirements:
```
pip install -r requirements.txt
```
To install in standard mode:
```
pip install .
```
To install in development mode:
```
pip install -e .
```

---
## Experiments
To run a default experiment, use:
```
python main.py
```
The default experiment will train the XX model on the Two-View MNIST dataset.

The code is written mainly with PyTorch (https://pytorch.org/).

The experiments are defined using the Hydra configs (https://hydra.cc/docs/next/intro) and located in the directory `configs`.

The training pipeline is based on the Catalyst framework (https://catalyst-team.github.io/catalyst/).

---
## Pre-trained models
The pre-trained models for OASIS 3 dataset can be downloaded using [this link](https://drive.google.com/file/d/1knfQGXq0G2hoEmcnOQsswm-lgd2IpUuJ/view?usp=sharing). These weights are under Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License (CC BY-NC-ND 4.0).

---
## Questions
If you have any questions about implementation and training, don't hesitate to either open an issue here or send an email to eidos92@gmail.com.


--- 
## Citing
If you find Fusion useful in your research, please use the following BibTeX entry for citation.
```
@article{fedorov2023self,
  title={Self-supervised multimodal learning for group inferences from MRI data: Discovering disorder-relevant brain regions and multimodal links},
  author={Fedorov, Alex and Geenjaar, Eloy and Wu, Lei and Sylvain, Tristan and DeRamus, Thomas P and Luck, Margaux and Misiura, Maria and Mittapalle, Girish and Hjelm, R Devon and Plis, Sergey M and others},
  journal={NeuroImage},
  pages={120485},
  year={2023},
  publisher={Elsevier}
}
```
---
## Acknowledgement
Specials thanks to [Devon Hjelm](https://github.com/rdevon) and [Philip Bachman](https://github.com/Philip-Bachman) for providing code for [DIM](https://github.com/rdevon/DIM) and [AMDIM](https://github.com/Philip-Bachman/amdim-public).

Additionally, thanks to [Sergey Kolesnikov](https://github.com/Scitator) for the help on [Catalyst](https://github.com/catalyst-team/catalyst) framework and [Kevin Wang](https://github.com/ssktotoro) for the support.

This work is supported by NIH R01 EB006841.

Data were provided in part by OASIS-3: Principal
Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P50
AG00561, P30 NS09857781, P01 AG026276, P01 AG003991,
R01 AG043434, UL1 TR000448, R01 EB009352. AV-45
doses were provided by Avid Radiopharmaceuticals, a
wholly-owned subsidiary of Eli Lilly.
