---
layout: post
title: Black-Scholes, Measured Carefully
date: 2023-03-01 12:00:00-0000
description: Going beyond the risk neutral probability distribution
tags: stochastic-calculus probability
categories: pricing
related_posts: false
---

### The long and winding road

For anyone who has ever gone through the trouble of deriving the [Black-Scholes
formula](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula)
from first principles, this process will be recognised as tedious and requiring
some attention to detail. This largely holds true, even if we assume the correctness of the 
[Black-Scholes partial differential equation](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_equation)
and the associated Feynman-Kac solution. Specifically, let us assume that the time zero value
of a call option can be expressed as the expectation of the discounted payoff,

\begin{equation}\label{call}
C_0 = \mathbb{E}^{\mathbb{Q}} [ e^{-rT} (S_T-K)^+ ] ,
\end{equation}

where $$\mathbb{Q}$$ is defined through the Radon-Nikodym derivative

$$
\left.{\frac{d \mathbb{Q}}{d \mathbb{P}}}\right|_t \equiv e^{ -\tfrac{1}{2} \lambda^2 t - \lambda W_t}, 
$$

and $$\lambda \equiv \frac{\mu-r}{\sigma}$$ is the market price of risk. We are still left with (i) determining the exact (non-relational) form risk-neutral
density and (ii) evaluating the resulting integrals. As for part (i), I personally prefer
taking the [unconscious approach](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician)
by first expressing the terminal value of the underlying as a function of a standard normal
random variable, $$Z \sim N(0,1)$$, viz.

$$
S_T(Z) = S_0 e^{ (r-\tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z },
$$
 
which follows by applying [Itô's Lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma#It%C3%B4_drift-diffusion_processes_(due_to:_Kunita%E2%80%93Watanabe)) to $$\ln S_t$$. 
The appropriate density to use is therefore the standard normal distribution, $$\phi(z)=\frac{e^{-z^2/2}}{\sqrt{2\pi}}$$.[^1] 
As for part (ii), we need to evaluate the integral

$$
C_0 = e^{-rT} \int_{\mathbb{R}} \left( S_T(z) -K \right)\boldsymbol{1}\{S_T(z) \geq K\}  \phi(z) dz,
$$

where $$\boldsymbol{1}\{ \cdot\}$$ is the indicator function. The $$K$$ term is trivially expressed through the cumulative normal distribution. On the other hand, the $$S_T(z)$$ term
can only be expressed as such after we first complete the square in the exponent and then perform
a change of variables. I won't reproduce the details here, but hopefully the reader
catches my drift. 

### What's your measure, sir?

What I am really trying to get at is the observation that a firm foundation in
probability theory will lead us to the Black-Scholes formula faster and more elegantly.
In fact, we can obtain the desired result without doing any integration whatsoever. 
The trick here is to think carefully about which probability measure we are using: in particular,
moving beyond the mindset that all valuation must be done exclusively under the risk neutral measure.

Recall that in the general no-arbitrage relation for contingent claims,

\begin{equation}\label{hi}
V_0 = \frac{V_0}{e^{r \cdot 0}} = \mathbb{E}^{\mathbb{Q}} \left[ \frac{V_T}{e^{rT}} \right],
\end{equation}

asset prices measured in units of the risk free asset ($$B_t=e^{rt}$$) are $$\mathbb{Q}$$-martingales.
Remarkably, it transpires that [equivalent probability measures](https://en.wikipedia.org/wiki/Equivalence_(measure_theory)) exist such that asset prices, 
measured in units of other tradeable assets, likewise are martingales. To see this, observe that the quantity

$$
e^{rt} \frac{S_t}{S_0} = e^{-\tfrac{1}{2} \sigma^2 t + \sigma W_t^{\mathbb{Q}}},
$$

is a positive $$\mathbb{Q}$$-martingale with unit expectation. Thus, [Girsanov's theorem](https://en.wikipedia.org/wiki/Girsanov_theorem)
tells us that we can define an equivalent measure $$\mathbb{Q}^S \sim \mathbb{Q}$$:

$$
\left.{\frac{d \mathbb{Q}^S}{d \mathbb{Q}}}\right|_t \equiv e^{-rt} \frac{S_t}{S_0},
$$

such that 

\begin{equation}\label{gs}
W_t^{\mathbb{Q}^S} = W_t^{\mathbb{Q}} - \sigma t. 
\end{equation}

We can therefore express \eqref{hi} as

$$
V_0 = \int \frac{V_T}{e^{rT}} d \mathbb{Q}_T = \int \frac{V_T}{e^{rT}} e^{rT} \frac{S_0}{S_T} d\mathbb{Q}_T^S = S_0 \int \frac{V_T}{S_T} d\mathbb{Q}_T^S,     
$$

or identically 

$$
\frac{V_0}{S_0} = \mathbb{E}^{\mathbb{Q}^S} \left[ \frac{V_T}{S_T} \right],
$$ 

which is what we wanted to show. As the unit of account has been changed, we refer to this process as a change of [numeraire](https://www.investopedia.com/terms/n/numeraire.asp).  

To see how this aids in the derivation of the Black-Scholes formula, let us write \eqref{call} as

$$
C_0 = \mathbb{E}^{\mathbb{Q}}[e^{-rT} S_T \boldsymbol{1}\{S_T \geq K\}] - e^{-rT}K \mathbb{E}^{\mathbb{Q}}[ \boldsymbol{1}\{S_T \geq K\}]. 
$$

The latter expectation is trivially identified as $$\mathbb{Q}(S_T \geq K)$$
i.e. the risk neutral probability of the option expiring in the money. Meanwhile, for the first term, it
is clearly advantageous to value the stock - not in units of the risk free asset - but rather in units of
the stock itself. In other words: to perform a change of measure from $$\mathbb{Q}$$ to $$\mathbb{Q}^S$$.
Doing so the first term becomes $$S_0 \mathbb{Q}^S(S_T \geq K)$$, which is the present value of the stock, weighted by the stock-measure probability
of the option expiring in the money. Jointly, we therefore have the fairly appealing pricing relation:

$$
C_0 = S_0 \mathbb{Q}^S(S_T \geq K) - e^{-rT}K\mathbb{Q}(S_T \geq K).
$$

Now observe that under the risk neutral measure $$\ln(S_T)$$ is normally distributed with mean $$m=\ln(S_0) + (r-\tfrac{1}{2}\sigma^2)T$$ and
variance $$v^2 = \sigma^2T$$. Letting $$Z$$ be a standard normal random variable we find that 

$$
\mathbb{Q}(S_T \geq K) = \mathbb{Q}(m + vZ \geq \ln(K)) = 1-\mathbb{Q}(Z \leq \tfrac{\ln(K)-m}{v}) = \Phi(d_2),
$$

where $$\Phi(\cdot)$$ is the cumulative normal distribution, and $$d_2$$ has the usual definition:

$$
d_2 \equiv \frac{\ln(S_0/K) + (r-\tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}.
$$

In a similar vein, under the stock measure $$\ln(S_T)$$ is normally distributed, this time with mean $$m=\ln(S_0) + (r+\tfrac{1}{2}\sigma^2)T$$ and
variance $$v^2 = \sigma^2T$$ (cf. eqn. \eqref{gs}).
This immediately allows us to deduce 

$$
\mathbb{Q}^S(S_T \geq K) = \Phi(d_1),
$$ 

where 

$$
d_1 \equiv d_2 + \sigma \sqrt{T}.
$$

Taken together, the Black-Scholes formula is manifest:

$$
C_0 = S_0 \Phi(d_1) - e^{-rT}K \Phi(d_2).
$$

### Parting thoughts

Emphatically, the power of measure/numeraire changes transcends the simple example provided here. 
Valuing an option on a coupon bearing bond? Consider using a zero-coupon bond as a numeraire.
Valuing a swaption? Using an annuity as a numeraire will make your life easier.
Plenty of such examples exist, and I encourage you to explore such possibilities when tasked with a valuation exercise. 



[^1]: Alternatively we could keep $$S_T$$ and work directly with the transition density $$d{\mathbb{Q}}(S_T \vert S_0)$$. To this end, we could solve the Fokker-Planck equation as demonstrated [here](https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Properties). I'll leave it as an exercise to the reader to show that the two approaches are in fact equivalent.


