# Keep Your Eye on the Best: Contrastive Regression Transformer for Skill Assessment in Robotic Surgery
Code for the paper "Keep Your Eye on the Best: Contrastive Regression Transformer for Skill Assessment in Robotic Surgery", published in IEEE Robotics and Automation Letters (RA-L).
To be presented in IROS 2023, Detroit, Michigan, USA.

### Introduction
We propose a novel video-based, contrastive
regression architecture, Contra-Sformer, for automated
surgical skill assessment in robot-assisted surgery. The proposed
framework is structured to capture the differences in the
surgical performance, between a test video and a reference video
which represents optimal surgical execution. A feature extractor
combining a spatial component (ResNet-18), supervised on
frame-level with gesture labels, and a temporal component
(TCN), generates spatio-temporal feature matrices of the test
and reference videos. These are then fed into an actionaware
Transformer with multi-head attention (A-Transformer)
that produces inter-video contrastive features at frame level,
representative of the skill similarity/deviation between the two
videos.

![Contra-Sformer](ContraSformer.png)
