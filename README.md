## An Active Learning Paradigm for Online Audio-Visual Emotion Recognition

This repository contains the **official code** for developing an online emotion recognition classifier using audio-visual modalities and deep reinforcement learning technigues introduced [here](https://ieeexplore.ieee.org/document/8937495).

Combined with corresponding repositories for preprocessing unimodal and multi-modal emotional datasets, like [AffectNet](http://mohammadmahoor.com/affectnet/), [IEMOCAP](https://sail.usc.edu/iemocap/), [RML](http://shachi.org/resources/4965), [BAUM-1](https://archive.ics.uci.edu/ml/datasets/BAUM-1), to produce the papers results.

Preprocessing codes for AffectNet, IEMOCAP and RML are provided by the authors, [here](https://github.com/IoannisKansizoglou/AffectNet-preprocess), [here](https://github.com/IoannisKansizoglou/Iemocap-preprocess) and [here](https://github.com/IoannisKansizoglou/RML-preprocess), respectively.

If you find this repository useful in your research, please consider citing:

    @article{kansizoglou2019active,
      title={An Active Learning Paradigm for Online Audio-Visual Emotion Recognition},
      author={Kansizoglou, Ioannis and Bampis, Loukas and Gasteratos, Antonios},
      journal={IEEE Transactions on Affective Computing},
      year={2019}
    }

Provided code is tested in Python 3.7.4 and Pytorch 1.4.0.

### Inputs Format

The ```params.json``` sets the training hyper-parameters, the exploited modality from the set ```{"audio", "visual", "fusion"}``` and the name of the speaker that is subtracted from the training dataset for evaluation. Note that *Leave-One-Speaker-Out* and *Leave-One-Speakers-Group_Out* schemes are adopted.

The following models are trained through two .csv files, including the paths of the training and evaluation samples, respectively. Those files shall be stored inside ```./data/speaker_folder```, where ```speaker_folder``` shall be given to the ```"speaker"``` variable in the ```params.json``` file.

### Usage


