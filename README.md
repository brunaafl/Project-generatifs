# Project based on the paper "Plug-and-Play Gibbs sampler: Embedding Deep Generative Priors in Bayesian Inference"

This project is based on the paper "Plug-and-Play Gibbs Sampler: Embedding Deep Generative Priors in Bayesian Inference." Our goal is to analyze and extend the framework by exploring different choices of priors and comparing their performance in image restoration tasks.

## Objectives
We focus on three key objectives:

**1. Reproducing the Paper's Results**
    * Implement and validate the Plug-and-Play (PnP) Gibbs sampler in the context of image deblurring using a Gaussian blur mask.
    * Ensure consistency with the results reported in the original paper.

**2. Comparing PnP Sampling with Direct DDPM Sampling:**
   
The PnP sampler generates samples from the posterior distribution by iteratively improving the estimate. This is done by sampling $x$ such that this is close to the denoised auxiliary variable $z$, and then
denoising $z$ to improve its quality.
We compare this with a simpler approach: directly sampling from a pre-trained Denoising Diffusion Probabilistic Model (DDPM) without the Gibbs framework.
This comparison will highlight the differences between explicit Bayesian inference and direct generative sampling, such as the possibility of generating distributions when using a Bayesian approach.

**3. Exploring Different Denoiser Priors:**
   
The original paper employs a diffusion-based denoiser as the prior model. However, it just requires that the prior $p(z∣x)$ acts as a denoiser. 
We investigate alternative denoising models, such as convolutional denoiser (DnCNN from Zhang et al., 2017), and Denoising autoencoder, which are less expensive to train and, therefore, interesting to be tested.

The modification to the PnP pipeline is minimal: At each Gibbs step $n$, you

  1. Sample $x^n \sim p(x|y,z)$
  2. Update $z^n \sim g(x)$, where $g$ is the generative model that produces a denoised estimate of $x$

Since $z$ becomes progressively less noisy at each Gibbs step, and $x$ is drawn to be close to $z$ this could lead to improved image reconstruction.
