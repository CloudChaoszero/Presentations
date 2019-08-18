# AB Testing in Python

![Logo](Resources/Images/Slides-Logo.png)
A PyBay 2019 talk
by Raul Maldonado

**Repo & Details Link**

http://bit.ly/Raul-ABTesting

**Google Colab**

[Problems here](https://drive.google.com/file/d/1XbyBPUUV9_5r89iZ9IATtQD23Jr0e9J8/view?usp=sharing)

[Solutions here](https://drive.google.com/file/d/1GHMO8xcVkdoi7Q0IYyel4c-eyiZ8K8Ue/view?usp=sharing)

## Overview

This repository is designated for my PyBay 2019 talk on an (frequentist) approach to A/B Testing, in Python.

Enjoy! :D
______

A/B Testing is "[a randomized experiment of two variants, A and B.](https://en.wikipedia.org/wiki/A/B_testing)” This test quantitatively compares two variants/samples with a single "metric of choice" in evaluation if there exists a statistical significance between said groups.


For example, let's say we ran a digital ad campaign A, with a Call to Action caption 'Click here, please!' 

Also, let's say we have an alternative ad campaign B with slight modification from A, being the change in the Call to Action to "Learn more here!". 

This in mind, our goal is to decide to see if there is a difference in the campaign's Click Through Rate (CTR) performance such that we increase our engagement.

> We define $\text{CTR} := \tfrac{\text{Total Number of Successes}}{\text{Total Number of Events}}$

That is, Considering two ad Campaigns A & B, each with it's unique distinction, we want to evaluate if there is a difference, a statistical difference in fact, in CTR performance.


![Clicks](https://media.giphy.com/media/3ogwG8ByATNb5EOm8E/giphy.gif)

## Resources

* [Code](Code/)

* [Technical Requirements](Resources/technicalRequirements/environment.yaml)
    * ```conda env create -f environment.yaml```
    * Yes, this is conda dependent :/

* [Slides](https://docs.google.com/presentation/d/1nr8O-hS070yhBZoc5KtMQgEdremG0-oZP0ujUnocELc/edit?usp=sharing)