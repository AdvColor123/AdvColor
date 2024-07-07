-- Disclaimer: This GitHub repository is under routine maintenance.
  
# AdvColor

This is the source code for AdvColor.

## Requirements
+ python == 3.7
+ pytorch == 1.13.0
+ torchvision == 0.14.0
+ tensorflow == 1.14.0
+ opencv

## Dataset

Please download the face anti-spoofing dataset [Replay-Attack](https://www.idiap.ch/en/scientific-research/data/replayattack).

## Training
We use this [repo](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch) to train the CDCN model for Replay-Attack. Replace the checkpoint in `main.py` after training.

## Usage

Run `python main.py` to launch the attack. Before that, remember to specify all the paths.

## Acknowledgement
* Face quality evaluation: [LightQNet](https://github.com/KaenChan/lightqnet).
* CDCN model: [CDCN](https://github.com/ZitongYu/CDCN).
* Training for CDCN: [CDCN-Face-Anti-Spoofing.pytorch](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch).