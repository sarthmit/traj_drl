## From Points to Functions: Infinite-dimensional Representations in Diffusion Models
___
This repository contains the official implementation for the paper **[From Points to Functions: Infinite-dimensional Representations in Diffusion Models](to-do)**


<details>
  <summary>
    <b>Abstract</b>
  </summary>
Diffusion-based generative models learn to iteratively transfer unstructured noise to a complex target distribution as opposed to Generative Adversarial Networks (GANs) or the decoder of Variational Autoencoders (VAEs) which produce samples from the target distribution in a single step. Thus, in diffusion models every sample is naturally connected to a random trajectory which is a solution to a learned stochastic differential equation (SDE). Generative models are only concerned with the final state of this trajectory that delivers samples from the desired distribution. Abstreiter et. al (2021) showed that these stochastic trajectories can be seen as continuous filters that wash out information along the way. Consequently, it is reasonable to ask if there is an intermediate time step at which the preserved information is optimal for a given downstream task. In this work, we show that a combination of information content from different time steps gives a strictly better representation for the downstream task. We introduce an attention and recurrence based modules that ``learn to mix'' information content of various time-steps such that the resultant representation leads to superior performance in downstream tasks.
</details>

---
Our work consists of analysis done on different types of datasets to see whether the trajectory based representations learned using score-based modeling leads to different information at different parts of the trajectory. In particular, we do analysis on standard benchmarks like CIFAR10 and CIFAR100 as well as different multi-task scenarios like Colored-MNIST and CelebA. The data used in the experiments can be found **[here](https://drive.google.com/file/d/13QIH8RUSxegJT7My-e9sCekaeZwkl-dc/view?usp=sharing)**.

---

We refer the readers to the respective sub-directories for details regarding each of the experiments. The code is heavily adapted from **[Yang Song's codebase](https://github.com/yang-song/score_sde)** 

* `diffusion_model`: Directory for training score-based models. In particular, we train the score model using `main.py` (alternatively run-file `run.sh`) after which we compute the representations using `representations.py` (alternatively run-file `rep.sh`).
* `downstream_classification`: Once the representations have been computed, we use the scripts in these directories to train MLP, transformer and other models for getting performance plots on CIFAR10/CIFAR100/Mini-ImageNet.
* `qualitative_analysis`: Finally, we use the code present here to perform experiments related to attention ablations, mutual information, activation profiles, etc.

Please do cite our work if you build up on it or find it useful and feel free to create an issue or contact me at `sarthmit@gmail.com` in case of any questions.

```
to-do
```