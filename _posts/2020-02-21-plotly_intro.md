---
title: "An Interactive Demo of Plotly (Plotly Express)"
date: 2020-02-12
tags: [visualization]
header:
    image: "/images/visualize.jpg"
excerpt: "Gentle introduction on how to use Plotly correctly for data visualization."
mathjax: "true"
---

# Plotly - what is it and why should YOU use it?

Plotly is helping leading organizations close the gap between Data Science teams and the rest of the organization. With Plotly, teams can easily design, develop, and operationalize data science initiatives that deliver real results. You can easily:

* Deploy data science and AI at scale.
* Move faster while achieving transparency.
* Develop data and model literacy across your organization.
* Reduce costs.
* Align technical and business teams.
* Maintain tight security.

Ar least that is what `plotly` promises on their [website](https://plot.ly/) - but what are my thoughts and why do I use plotly?

Basically when I talk about ploty I mainly talk about the beautiful `plotly.express` part of plotly. It is just the great and easy syntax (which is really really really similar to seaborn) and the interactive component which sets `plotly.express` apart from anything else. Of course I know and also use the *standard* functionalities of plotly or create some dashboards with dash, but especially for data exploration everrybody should at least know about the power `plotly.express` delivers out of the box.

I used `matplotlib` (and never really liked it) until I found out about `seaborn` which kind of wrappes the sometimes complicated syntax to most of the times one line of code. Having this the step to plotly.express was so easy that I basically completly erased `matplotlib` and `seaborn` out of my daily workflow.

But enough about my prefered workflow, let's get our hands dirty, checkout out why plotly is that awesome and learn how to use it!

## Set-Up

First things first, I assume that you already know how to use `env's` and are familiar with [anaconda](https://anaconda.org/). If not, I have some great links for you:
* https://towardsdatascience.com/get-your-computer-ready-for-machine-learning-how-what-and-why-you-should-use-anaconda-miniconda-d213444f36d6
* https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533

Now we will create an conda env which contains everything we need for this tutorial: `plotly and pandas`.

```bash
$ conda create -n plotly python=3.6 pandas plotly
$ conda activate plotly
```

Let's fire up our notebook and start out data exploration:

```bash
jupyter notebook
```

## Plotly Express

