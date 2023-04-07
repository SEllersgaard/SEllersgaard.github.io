---
layout: post
title: Model Multiplicity  
date: 2023-04-04 12:00:00-0000
description: Incorporating model uncertainty into your machine learning pipeline
tags: machine-learning probability
categories: modelling 
related_posts: false
---

{% include figure.html path="assets/img/uncertain2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
*Probabilistic uncertainty as interpreted by DALL-E 2*

### Bayesian model averaging  
                                         
A well-known aphorism attributed to [George Box](https://en.wikipedia.org/wiki/George_E._P._Box) states that "*all models 
are wrong, but some are useful*". This observation rings particularly true in systematic investing, where the price action
governed by millions of non-cooperating players, ultimately is explained through some sort of reductionist supervised learning 
problem. Nobody would seriously entertain the view that these models are *fundamental* or *expressively adequate*, 
yet it is common practice to let just a single model guide the trading process based on some performance metric like the
out-of-sample mean-squared error. This is potentially troubling given that a less accurate model need not be universally
inferior in all of its predictions. Indeed, it is conceivable that some biases cancel each other out if only we marry the predictions made by various models (see e.g. the work by [Jennifer Hoeting](https://en.wikipedia.org/wiki/Jennifer_A._Hoeting)).
This begs the question: how exactly do we combine said predictions? Intuitively, while simple linear averaging might work,
the thought of attributing equal significance to models regardless of their level of performance is clearly distasteful. A somewhat subtler approach would be to weigh a prediction by the evidence of the model, and this is precisely what
*Bayesian model averaging* sets out to do. In this piece I will provide an exegesis of this philosophy, and lay out what I
consider to be serious obstacles and how to potentially overcome them. 

Let $$\boldsymbol{y}=(y_1, y_2, ..., y_N)^\intercal \in \mathbb{R}^{N}$$ be a vector of data observations, which we desire to model. 
Furthermore, suppose we have $$K$$ competing models $$\mathbb{M} = \{M_1, M_2, ..., M_K \}$$ for $$\boldsymbol{y}$$, each of which is characterised by some specific vector of parameters: $$\boldsymbol{\theta}_i = 
(\theta_{i,1},\theta_{i,2}, ..., \theta_{i,T_i})^\intercal \in \boldsymbol{\Theta}_i \subseteq \mathbb{R}^{T_i}$$.
The candidate models could be [nested](https://www.theanalysisfactor.com/what-are-nested-models/)
within the same super-model (e.g. all possible subsets of a multivariate linear regression), although this is *not* a requirement.  

Based on observable evidence, what is the probability of any given model? According to [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
it follows that $$\forall i$$:

\begin{equation}\label{bayes}
p(M_i | \boldsymbol{y} ) = \frac{p(\boldsymbol{y}| M_i) p(M_i)  }{\sum_{j=1}^K p(\boldsymbol{y}| M_j) p(M_j)},
\end{equation}

where $$p(\boldsymbol{y} \vert M_i)$$ can be written as 

\begin{equation}\label{bayes2}
p(\boldsymbol{y}| M_i) = \int_{\boldsymbol{\Theta}_i} p(\boldsymbol{y}| \boldsymbol{\theta}_i, M_i) p(\boldsymbol{\theta}_i | M_i) d\boldsymbol{\theta}_i.                                                                                         
\end{equation}

Here, the probability $$p(M_i)$$ signifies our credence in model $$M_i$$ before being presented with any available data. Obviously,
for any given model, this is an entirely subjective matter, although a popular choice unsurprisingly is that of [uniformity](https://en.wikipedia.org/wiki/Prior_probability#Uninformative_priors):
$$\forall i: p(M_i)=1/K$$. In a similar vein, the probability $$p(\boldsymbol{\theta}_i | M_i)$$ codifies 
our prior beliefs about how the parameters of model $$M_i$$ are distributed, assuming the correctness of that model: again, a
matter inherently subjective in nature. While people anchored in the frequentists' paradigm might find the concept of subjective priors disturbingly nebulous, let
me assure you that the importance of priors often is *over-stated*. Specifically, for sufficiently large $$N$$ (no. of observations), sufficiently diffuse priors 
will be overshadowed by the evidence leading to approximately identical posteriors. The important point is that 
equation \eqref{bayes} provides a coherent framework for going from no empirical evidence, to looking at the data, to ultimately
assigning an evidence based probability score to a given model. 

Now we can use our collection of posterior probabilities over the space of models $$\mathbb{M}$$ to get a sense
of "model free" (strictly speaking: model weighted) probabilities: e.g. if $$\Delta$$ is some quantity of interest then
$$p(\Delta | \boldsymbol{y}) = \sum_{j=1}^K p(\Delta | M_j, \boldsymbol{y}) p(M_j | \boldsymbol{y}).$$ E.g. if we are in 
the business of forecasting, we can get a model-averaged expectation of the next observation using the equation:

\begin{equation}\label{expectation}
\mathbb{E}[y^* | \boldsymbol{y}] = \sum_{j=1}^K \mathbb{E}[y^* | \boldsymbol{y}, M_j] p(M_j | \boldsymbol{y}).  
\end{equation}    

Great, so should we start throwing everything but the kitchen sink when modelling our data? Well, not quite...

### The Schwarz approximation

There's a rather glaring issue with \eqref{bayes} having to do with computability. For starters, the space of possible models 
quickly grows almost unimaginably large. For instance, the number of possible linear regressions you can run with 50
input features is well over a quadrillion ($$10^{15}$$). Secondly, finding explicit expression for the posterior distribution
may often prove onerous if not downright impossible: instead, computer-intensive numerical methods such as [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
(MCMC) must be called up. If this in turn must be run repeatedly in a back-testing system, the computational challenges quickly
become insurmountable. 

Neither issue is trivially resolved, but there are a few tricks of the trade commonly employed. E.g. drastic reductions in
the model space cardinality are quickly accomplished by (a) excluding models which predict the data far less well than the best
model, and (b) throwing out complex models that receive less support from the data, than simpler models [[Hoeting et al.](https://www.jstor.org/stable/2676803)].
Meanwhile, to circumvent the MCMC issue, we may follow Gideon Schwarz in approximating the posterior distribution using
[Laplace's method](https://en.wikipedia.org/wiki/Laplace%27s_method).

The basic idea is as follows[^1]: suppose we take a flat prior on $$\boldsymbol{\Theta}_i$$. Furthermore, let $$\ell(\boldsymbol{\theta}) \equiv \log(p(\boldsymbol{y}| \boldsymbol{\theta}_i, M_i) )$$
denote the log-likelihood, $$\bar{\ell}$$ the mean log-likelihood, and let $$\hat{\boldsymbol{\theta}}_i$$ be the maximum likelihood estimator (MLE), then
\eqref{bayes2} can be written as

$$
\begin{aligned}
p(\boldsymbol{y} | M_i) &\propto \int_{\boldsymbol{\Theta}_i} e^{N \bar{\ell}(\boldsymbol{\theta}_i)} d\boldsymbol{\theta}_i \\
& \approx e^{\ell(\hat{\boldsymbol{\theta}}_i)} \int_{\boldsymbol{\Theta}_i} e^{-\tfrac{1}{2} (\boldsymbol{\theta}_i - \hat{\boldsymbol{\theta}}_i)^\intercal N\frac{\partial^2 \bar{\ell}(\hat{\boldsymbol{\theta}}_i) }{\partial \boldsymbol{\theta}_i \partial \boldsymbol{\theta}_i^\intercal} (\boldsymbol{\theta}_i - \hat{\boldsymbol{\theta}}_i)} d\boldsymbol{\theta}_i \\
& \approx e^{\ell(\hat{\boldsymbol{\theta}}_i)} (2 \pi)^{-T_i/2} N^{-T_i/2} \det \left(\frac{\partial^2 \bar{\ell}(\hat{\boldsymbol{\theta}}_i) }{\partial \boldsymbol{\theta}_i \partial \boldsymbol{\theta}_i^\intercal} \right)^{1/2},
\end{aligned}
$$

where the second line uses a [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series) around the MLE, and the third
line executes the [multivariate Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral#n-dimensional_and_functional_generalization). According to Schwarz this
can be further simplified for large $$N$$ as

$$
p(\boldsymbol{y} | M_i) \propto e^{\ell(\hat{\boldsymbol{\theta}}_i)} N^{-T_i/2} = e^{-\tfrac{1}{2}\text{BIC}(M_i)},
$$

where $$\text{BIC}(M_i) \equiv -2 \ell(\hat{\boldsymbol{\theta}}_i)) + T_i \log(N)$$ is the [Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) -
a common in-sample measure used in identifying the "goodness of fit" of a model balanced against its complexity. Plugging this into \eqref{bayes} and taking a flat
prior on the model space $$\mathbb{M}$$ we finally arrive at the following expression for data-driven model probabilities: 

\begin{equation}\label{pm}
p(M_i | \boldsymbol{y}) = \frac{e^{-\tfrac{1}{2}\text{BIC}(M_i)}}{ \sum_{j=1}^K e^{-\tfrac{1}{2}\text{BIC}(M_j)}}.
\end{equation}

Simply put: when all models have equal complexity we simply weigh their predictions based on their likelihood function.

### Towards Generality

If practitioners of data science are put off by Bayesian information criterions, likelihood functions etc. I wouldn't hold
it against them. While these concepts are prevalent within statistics, they are less ubiquitous amongst computer scientists: 
indeed, they may not even be well-defined for many of the machine learning models commonly employed today. 
Further work is therefore warranted. Here, I'll present a [Wittgenstein's ladder](https://en.wikipedia.org/wiki/Wittgenstein%27s_ladder)-type
argument for what I think ought to be done. 

> "*My propositions serve as elucidations in the following way: anyone who understands me eventually recognizes them as nonsensical, 
when he has used them—as steps—to climb beyond them. He must, so to speak, throw away the ladder after he has climbed up it*".

Suppose we have some model $$M_i: \boldsymbol{y}_t = f(\boldsymbol{x}_t \vert \boldsymbol{\theta}_i) + \varepsilon_t$$ for $$t=1,2,...,N$$ where
$$\varepsilon_t \sim \mathcal{N}(0,\sigma_i^2)$$ is an i.i.d. error term. The likelihood function for this model can be written as

$$
L(\boldsymbol{\theta}_i) = \prod_{t=1}^N \frac{ e^{-\frac{(y_t - f(\boldsymbol{x}_t \vert \boldsymbol{\theta}_i))^2}{2\sigma_i^2}}}{\sqrt{2 \pi \sigma_i^2}},
$$

or in log-likelihood terms:

$$
\ell(\boldsymbol{\theta}_i) = - \frac{N}{2} \ln(2 \pi) - \frac{N}{2} \ln(\sigma_i^2) - \frac{1}{2\sigma_i^2} RSS_i,
$$

where $$RSS_i$$ is the [residual sum of sqaures](https://en.wikipedia.org/wiki/Residual_sum_of_squares). 
Now $$\sigma_i^2 \approx RSS_i/N = MSE_i$$ (the mean squared error) so this expression boils down to

$$
\ell(\boldsymbol{\theta}_i) = -\frac{N}{2} \ln(MSE_i) + \text{terms depending on }N.
$$

Hence the Bayesian information criterion can be written as

$$
BIC(M_i) = \frac{N}{2} \ln(MSE_i) + T_i \log(N).
$$

This offers a somewhat more appealing way of writing \eqref{pm}, with the caveat that we still have to deal with 
quantifying model complexity (the $$T_i$$ penalty term). Now in practice what I would do instead is the following: It is well known that minimising the BIC asymptotically 
is equivalent to leave-of-$$\nu$$ cross-validation for linear models (see [this reference](https://robjhyndman.com/hyndsight/crossvalidation/)).
This suggests the following approach: for each model tune hyper-parameters using cross-validation.
Rather than weighing predictions of the tuned models by their $$BIC$$-score, let's weigh them by their cross-validated mean square error. 
Something as simple as

\begin{equation}\label{pm2}
p(M_i | \boldsymbol{y}) = \frac{ MSE_{i,cv}^{-1} }{ \sum_{j=1}^K MSE_{j,cv}^{-1}},
\end{equation}

could do. The benefits of this are as follows: (a) it is extremely simple to calculate, and (b) no unfair advantage is given to over-fitting
models. 

Again note that this argument is purely heuristic in nature. I welcome alternative suggestions to this intensely fascinating subject. 







[^1]: For a more careful derivation I recommend Bhat and Kumar's [On the derivation of the Bayesian Information Criterion](https://faculty.ucmerced.edu/hbhat/BICderivation.pdf).






