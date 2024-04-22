# Regenerating Arbitrary Video Sequences with Distillation Path-Finding
by [Thi-Ngoc-Hanh Le](https://lehanhcs.github.io/), Sheng-Yi Yao, Chun-Te Wu and [Tong-Yee Lee](http://graphics.csie.ncku.edu.tw/). [Computer Graphics Group](http://graphics.csie.ncku.edu.tw/) at National Cheng-Kung University. <br>
Training code will be updated. <br>

This resposity is the official implementation of our video resequencing system. This paper has been accepted for publication on IEEE Transactions on Visualization and Computer Graphics (Jan/2023). <br>

Paper
---
* Published online on IEEE Transactions on Visualization and Computer Graphics, [link](https://ieeexplore.ieee.org/abstract/document/10018537)
* Project website: [link](http://graphics.csie.ncku.edu.tw/SDPF)
* Paper [*.pdf](TVCG_Video_Resequencing.pdf)

Introduction
---
If the video has long been mentioned as a widespread visualization form, the animation sequence in the video is mentioned as storytelling for people. Producing an animation requires intensive human labor from skilled professional artists to obtain plausible animation in both content and motion direction, incredibly for animations with complex content, multiple moving objects, and dense movement. We first learn the feature correlation on the frameset of the given video through a proposed network called **RSFNet**. Then, we develop a novel path-finding algorithm, **SDPF**, which formulates the knowledge of motion directions of the source video to estimate the smooth and plausible sequences.

![framework](https://github.com/LeHanhcs/SDPF_VideoResequencing/assets/37010753/027b6ea9-60e0-483a-8626-054c6c75d28d)

Requirements
---
tensorflow==1.15
tensorflow-estimator==1.15.1
keras==2.3.1
matplotlib
scikit-image
pandas
tqdm
opencv
scikit-learn
conda-forge::imageio-ffmpeg

Guidance of data/code usage
---
> python simulate.py –m 1 –s 0
```
-n: model name, in the paper is Tri_Fuse_AE
-s: user-specified initial frame (starting frame), 0 represents the 0th frame in the input video (or a bunch of pictures)
```

* Input video: please download on our [project website](http://graphics.csie.ncku.edu.tw/SDPF) 
* Extract input video to folder './data/image'. This folder will consist of splited frames from input video, optical flow files.
* Set parameter 'start_frame' to define the starting frame of the new sequence.
* Using [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://github.com/philferriere/tfoptflow) to generate optical flow of the input video.

Acknowledgments
---
Thanks for the authors of [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://github.com/philferriere/tfoptflow) for the code of generating optical flow. <br>
We inherit the code from their responsitory for optical flow production in our system.

Citation
---
If our method is useful for your research, please consider citing:
```
@article{yao2023regenerating,
  title={Regenerating Arbitrary Video Sequences with Distillation Path-Finding},
  author={Le, Thi-Ngoc-Hanh and Yao, Sheng-Yi and Wu, Chun-Te and Lee, Tong-Yee},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```

Contact
---
If you have any question, please email me: ngochanh.le1987@gmail.com or tonylee@mail.ncku.edu.tw (corresponding author)
