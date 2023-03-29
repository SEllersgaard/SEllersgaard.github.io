---
layout: post
title: Sub-optimal Optimal Portfolios  
date: 2023-03-27 12:00:00-0000
description: Exploring the effect of parameter uncertainty in optimal asset allocation
tags: econometrics trading
categories: portfolio
related_posts: false
---

### A perfect Brownian

Imagine you are truthfully told about the functional form of the stochastic equations of motion characterising
some set of stocks, with the caveat that you are required to estimate the associated parameters yourself. *Prima facie*,
this complete removal of [Knightian uncertainty](https://en.wikipedia.org/wiki/Knightian_uncertainty) sounds
like a dream scenario which trivially will allow for what we vaguely can call "perfect trading". 
In this piece I want to convince you that things are not quite so straight-forward. In particular, even
with a substantial amount of data at your disposal, parameter uncertainty can still undermine theoretically
optimal strategies. To cement this point, I will consider the simulated performance of a number of results from
[modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory), both in the event that we
*do* and *do not* have complete information about the input parameters. In a certain sense, this analysis is befitting considering that 
the father of modern portfolio theory,
[Harry Markowitz](https://en.wikipedia.org/wiki/Harry_Markowitz), famously [doesn't practice what he preaches](https://onlinelibrary.wiley.com/doi/10.1002/scin.5591791221),
being instead partial towards "equal weight" diversification.   


To set the scene, let us suppose that the daily percentage returns of $$K$$ assets are multivariate normal,
$$\boldsymbol{R}_t \sim \mathcal{N}(\boldsymbol{\mu} \Delta t, \boldsymbol{\Sigma} \Delta t)$$,
where $$\boldsymbol{\mu} \in \mathbb{R}^K$$ is an annualised mean return vector, and $$\boldsymbol{\Sigma} \in \mathbb{R}^{K \times K}$$
is an annualised covariance matrix i.e. $$\boldsymbol{\Sigma} \equiv \text{diag}(\boldsymbol{\sigma})\boldsymbol{\varrho}\text{diag}(\boldsymbol{\sigma}) $$
where $$\boldsymbol{\sigma} \in \mathbb{R}^K$$ is a volatility vector, and $$\boldsymbol{\varrho} \in \mathbb{R}^{K \times K}$$ is a correlation matrix. 
$$\Delta t$$ scales these quantities into daily terms and is typically set to 1/252 to reflect the average number of 
trading days in a year.

Equivalently, we may write this in price process terms as the dynamics

$$
\boldsymbol{S}_{t+1} = \boldsymbol{S}_{t} + \text{diag}(\boldsymbol{S}_{t})(\boldsymbol{\mu} \Delta t + \boldsymbol{L} \sqrt{\Delta t} \boldsymbol{Z}),
$$

where $$\boldsymbol{L}$$ is the lower triangular matrix arising from [Cholesky decomposing](https://en.wikipedia.org/wiki/Cholesky_decomposition)
$$\boldsymbol{\Sigma}$$, and $$\boldsymbol{Z}: \Omega \mapsto \mathbb{R}^K$$ is an i.i.d. standard normal vector i.e $$\boldsymbol{Z} \sim \mathcal{N}(\boldsymbol{0}, \mathbb{I}_{K})$$.

For the purpose of this exercise we shall consider five co-moving assets with the following parameter specifications:

```python
μ = np.array([0.1, 0.13, 0.08, 0.09, 0.14]) #mean
σ = np.array([0.1, 0.15, 0.1, 0.1, 0.2]) #volatility
ρ = np.array([[1, 0, 0, 0, 0],
              [0.2, 1, 0, 0, 0],
              [0.3, 0.5, 1, 0, 0],
              [0.1, 0.2, 0.1, 1, 0],
              [0.4, 0.1, 0.15, 0.2, 1]])
ρ = ρ+ρ.T-np.identity(len(μ)) #correlation
dt = 1/252 #1/trading days per year
T = 10 #simulation time [years] 
r = 0 #risk-free rate 
S_0 = 100*np.ones(len(μ)) #initial share prices
W_0 = 1e6 #initial welath
```
Furthermore, it will be convenient to define the following quantities:
```python
M = len(μ)
N = int(T/dt)
Σ = np.diag(σ)@ρ@np.diag(σ)
Σinv = np.linalg.inv(Σ)
sqdt = np.sqrt(dt)
λ, Λ = np.linalg.eig(Σ)
assert (λ > 0).all() #assert positive semi-definite
L = np.linalg.cholesky(Σ)
I = np.ones(M)
Sharpe = (μ-r)/σ
```
Finally, to simulate stock paths according to these inputs we can use the following vectorised function: 
```python
def multi_brownian_sim(seed: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if seed:
        np.random.seed(42)
    dW = pd.DataFrame(np.random.normal(0, sqdt, size=(M, N+1)))
    ret = (1 + L@dW + np.tile(μ*dt, (N+1,1)).T)
    ret = ret.T
    ret.iloc[0] = S_0
    df = ret.cumprod() #levels
    dfr = (df.diff()/df.shift(1)).dropna() #returns 

    return df, dfr
```
A single run of this yields something along the lines below:  

{% include figure.html path="assets/img/brownian.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

### Estimating parameters

Alright, suppose you are handed the data for these paths above and informed that returns are multivariate normal.
What do you conclude? Well, you'll probably find the unbiased estimators $$ \{\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\sigma}}, \hat{\boldsymbol{\varrho}} \}$$ 
as displayed in the 2nd column in the tables below. The noteworthy part here is how much some of these appear amiss 
vis-à-vis their population counterparts. So much so that one *almost
suspects* that something is messed up with the data-generating process per se. To assure you this is not the case, we will do a couple
of exercises: first, we will compute two-sigma confidence intervals for our estimators, and second, we will sample from the data-generating
process repeatedly and show that the averages over the estimators indeed are congruent with the true underlying dynamics.

Confidence intervals for $$\hat{\boldsymbol{\mu}}$$ and $$\hat{\boldsymbol{\sigma}}^2$$ can be computed using the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem): indeed,
it is straightforward to show that $$\sqrt{N}(\hat{\mu}_i - \mu_i) \sim \mathcal{N}(0, \sigma_i^2)$$ and $$\sqrt{N}(\hat{\sigma}_i^2 - \sigma_i^2) \sim \mathcal{N}(0, 2\sigma_i^4)$$
where $$N$$ is the number of observations.
Deploying the [delta method](https://en.wikipedia.org/wiki/Delta_method) to extract the limiting distribution of $$\hat{\boldsymbol{\sigma}}$$ we find that
$$\sqrt{N}(\hat{\sigma}_i - \sigma_i) \sim \mathcal{N}(0, \sigma_i^2/2)$$. Finally, confidence intervals for $$\hat{\varrho}_{ij}$$ can be estimated
using the [Fisher transformation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation).

For the Monte Carlo simulation we run the data generating process 1000 times storing the estimators for $$ \{\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\sigma}}, \hat{\boldsymbol{\varrho}} \}$$
from each run in order to ultimately study their statistical properties.
Per definition, 95% of the confidence intervals we'd compute from repeated sampling should encompass the true population parameters. 
Notice the expected values (6th column) across these samples align much more closely with the true parameters: 
a small victory for having the equivalent of 10,000 years' worth of financial data(!)



<table border="1" class="dataframe" width="100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$$\mu$$</th>
      <th>$$\hat{\mu}$$</th>
      <th>$$\hat{\mu}-2SE$$</th>
      <th>$$\hat{\mu}+2SE$$</th>
      <th>$$\in CI$$</th>
      <th>$$\mathbb{E}(\mu_{mc})$$</th>
      <th>$$\min(\mu_{mc})$$</th>
      <th>$$\max(\mu_{mc})$$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.10</td>
      <td>0.152</td>
      <td>0.090</td>
      <td>0.214</td>
      <td>True</td>
      <td>0.097</td>
      <td>0.003</td>
      <td>0.191</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>0.096</td>
      <td>-0.000</td>
      <td>0.192</td>
      <td>True</td>
      <td>0.131</td>
      <td>-0.035</td>
      <td>0.252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.08</td>
      <td>0.039</td>
      <td>-0.024</td>
      <td>0.103</td>
      <td>True</td>
      <td>0.080</td>
      <td>-0.027</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.09</td>
      <td>0.105</td>
      <td>0.043</td>
      <td>0.168</td>
      <td>True</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.14</td>
      <td>0.132</td>
      <td>0.007</td>
      <td>0.257</td>
      <td>True</td>
      <td>0.139</td>
      <td>-0.039</td>
      <td>0.339</td>
    </tr>
  </tbody>
</table>


<br>


<table border="1" class="dataframe" width="100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$$\sigma$$</th>
      <th>$$\hat{\sigma}$$</th>
      <th>$$\hat{\sigma}-2SE$$</th>
      <th>$$\hat{\sigma}+2SE$$</th>
      <th>$$\in CI$$</th>
      <th>$$\mathbb{E}(\sigma_{mc})$$</th>
      <th>$$\min(\sigma_{mc})$$</th>
      <th>$$\max(\sigma_{mc})$$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.10</td>
      <td>0.098</td>
      <td>0.096</td>
      <td>0.101</td>
      <td>True</td>
      <td>0.10</td>
      <td>0.096</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.15</td>
      <td>0.152</td>
      <td>0.147</td>
      <td>0.156</td>
      <td>True</td>
      <td>0.15</td>
      <td>0.144</td>
      <td>0.158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.10</td>
      <td>0.101</td>
      <td>0.098</td>
      <td>0.104</td>
      <td>True</td>
      <td>0.10</td>
      <td>0.096</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.10</td>
      <td>0.099</td>
      <td>0.096</td>
      <td>0.101</td>
      <td>True</td>
      <td>0.10</td>
      <td>0.095</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.20</td>
      <td>0.198</td>
      <td>0.192</td>
      <td>0.203</td>
      <td>True</td>
      <td>0.20</td>
      <td>0.191</td>
      <td>0.208</td>
    </tr>
  </tbody>
</table>


<br>


<table border="1" class="dataframe" width="100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>$$\rho$$</th>
      <th>$$\hat{\rho}$$</th>
      <th>$$\hat{\rho}-2SE$$</th>
      <th>$$\hat{\rho}+2SE$$</th>
      <th>$$\in CI$$</th>
      <th>$$\mathbb{E}(\rho_{mc})$$</th>
      <th>$$\min(\rho_{mc})$$</th>
      <th>$$\max(\rho_{mc})$$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0</th>
      <th>1</th>
      <td>0.20</td>
      <td>0.208</td>
      <td>0.169</td>
      <td>0.245</td>
      <td>True</td>
      <td>0.200</td>
      <td>0.146</td>
      <td>0.260</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.30</td>
      <td>0.292</td>
      <td>0.256</td>
      <td>0.328</td>
      <td>True</td>
      <td>0.299</td>
      <td>0.233</td>
      <td>0.365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.10</td>
      <td>0.057</td>
      <td>0.017</td>
      <td>0.096</td>
      <td>False</td>
      <td>0.100</td>
      <td>0.039</td>
      <td>0.159</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.40</td>
      <td>0.405</td>
      <td>0.371</td>
      <td>0.438</td>
      <td>True</td>
      <td>0.400</td>
      <td>0.340</td>
      <td>0.456</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>2</th>
      <td>0.50</td>
      <td>0.483</td>
      <td>0.452</td>
      <td>0.513</td>
      <td>True</td>
      <td>0.500</td>
      <td>0.447</td>
      <td>0.545</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.20</td>
      <td>0.189</td>
      <td>0.151</td>
      <td>0.227</td>
      <td>True</td>
      <td>0.200</td>
      <td>0.137</td>
      <td>0.256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.10</td>
      <td>0.138</td>
      <td>0.099</td>
      <td>0.177</td>
      <td>True</td>
      <td>0.100</td>
      <td>0.037</td>
      <td>0.163</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>3</th>
      <td>0.10</td>
      <td>0.124</td>
      <td>0.085</td>
      <td>0.163</td>
      <td>True</td>
      <td>0.101</td>
      <td>0.043</td>
      <td>0.164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.15</td>
      <td>0.179</td>
      <td>0.140</td>
      <td>0.217</td>
      <td>True</td>
      <td>0.150</td>
      <td>0.093</td>
      <td>0.208</td>
    </tr>
    <tr>
      <th>3</th>
      <th>4</th>
      <td>0.20</td>
      <td>0.199</td>
      <td>0.160</td>
      <td>0.237</td>
      <td>True</td>
      <td>0.200</td>
      <td>0.139</td>
      <td>0.261</td>
    </tr>
  </tbody>
</table>
<br>
Scanning through the above numbers the most problematic estimator is clearly $$\hat{\boldsymbol{\mu}}$$.
Intuitively, we can explain this as follows: since $$R_t \equiv (S_t-S_{t-1})/S_{t-1} \approx \ln(S_t)-\ln(S_{t-1})$$
it follows that 

$$
\bar{R} \equiv N^{-1} \sum_{i=1}^N R_t \approx N^{-1} \left( \ln(S_T)-\ln(S_{0}) \right),
$$

i.e. the estimator is a [telescoping series](https://en.wikipedia.org/wiki/Telescoping_series) effectively determined 
by the first and last entry. For "shorter" histories this can be extremely problematic (noisy). 

### The final frontier?

To get a sense of the sheer amount of gravity these uncertainties carry, it is helpful to consider the 
[efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier) i.e. the collection of portfolio constructions
which deliver the lowest amount of volatility for a given level of return. A well known result in modern portfolio theory
states that this frontier carves out a hyperbola in (standard deviation, mean)-space given by the equation

$$
\sigma_{\pi}^2(\mu_{\pi}) = \frac{C \mu_{\pi}^2 - 2B \mu_{\pi} + A}{D},
$$

where $$A \equiv \boldsymbol{\mu}^\intercal \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$$, $$B \equiv \boldsymbol{\mu}^\intercal \boldsymbol{\Sigma}^{-1} \boldsymbol{1}$$,
$$C \equiv \boldsymbol{1}^\intercal \boldsymbol{\Sigma}^{-1} \boldsymbol{1}$$, and $$D \equiv AC-B^2$$.

The function below allows us to visualise the efficient frontier for arbitrary $$\boldsymbol{\mu}$$, $$\boldsymbol{\Sigma}$$:

```python
def plot_frontier(μ_est: np.array, Σ_est: np.array, ax = None, scatter: bool = False, muc: float = 0.3, alpha: float = 0.5):

    σ_est = np.sqrt(np.diag(Σ_est))
    Σinv_est = np.linalg.inv(Σ_est)

    A = μ_est@Σinv_est@μ_est
    B = μ_est@Σinv_est@I
    C = I@Σinv_est@I
    D = A*C - B**2
    if μ_est.max() > muc:
        muc = μ_est.max() + 0.1
    mubar = np.arange(0,muc,0.005)
    sigbar = np.sqrt((C*pow(mubar,2) - 2*B*mubar + A)/D)

    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(13,11))

    ax.plot(sigbar, mubar, alpha=alpha)
    if scatter:
        ax.scatter(σ_est, μ_est)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Efficient Frontier')
```

E.g. for the specified parameters characterising the five assets, this yields the following curve:  

{% include figure.html path="assets/img/frontier1.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

Now let's consider what happens if $$\boldsymbol{\mu}$$ is known *a priori*, but $$\boldsymbol{\Sigma}$$ must
be estimated from the data. As above, we'll repeat the data generating process 1000 times, recomputing the frontier on each iteration, to get a sense of the variability.

{% include figure.html path="assets/img/frontier2.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

Evidently, while this introduces some estimation error, the overall picture remains "within reason". 

Finally, suppose both $$\boldsymbol{\mu}$$ are $$\boldsymbol{\Sigma}$$ are inferred *a posteriori*. The Monte Carlo run yields:  

{% include figure.html path="assets/img/frontier3.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

... which isn't exactly great. 

The question remains: to what extent does this actually impact quantitative trading strategies? Well, the efficient
frontier does include a couple of portfolios of interest, most notably the *minimum variance portfolio* and
the *maximum Sharpe ratio portfolio*, both of which plausibly could be adopted by practitioners. Let's see
how such strategies fare in light of our current framework. 

### One-period "optimal" trading strategies

One of the great tragedies of academic finance is the curious disconnect with which it operates from the practical field. 
Optimisation, for example, is a major division of mathematical finance, yet empirical tests thereof are shockingly scarce. 
So much so that the *enfant terrible* of the quant industry, Nassim Taleb, noted in "The Black Swan": 

> *"I would not be the first to say that this optimisation set back the social science by reducing it from
the intellectual and reflective discipline it was becoming to an attempt at an “exact science”. By
“exact science”, I mean a second-rate engineering problem for those who want to pretend that they
are in a physics department - so-called physics envy. In other words, an intellectual fraud".*

To remedy this deplorable situation, let us finally explore how optimal one-period portfolio constructions fare against equal allocation ($$1/K$$ diversification).
In particular, let us consider whether (I) the minimum variance portfolio[^1]:

$$
\boldsymbol{\pi}_{\text{minvar}} = \frac{\boldsymbol{\Sigma}^{-1} \boldsymbol{1}}{ \boldsymbol{1}^\intercal \boldsymbol{\Sigma}^{-1} \boldsymbol{1}},
$$

actually delivers a smaller variance than the $$1/K$$ benchmark. For reference: $$ \boldsymbol{\pi}_{\text{minvar}} \approx (0.32,  0.00,  0.31 ,  0.38, -0.01)$$.
Similarly whether (II) the maximum sharpe ratio portfolio[^2]:

$$
\boldsymbol{\pi}_{\text{maxshp}} = \frac{\boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}-r\boldsymbol{1})}{ \boldsymbol{1}^\intercal \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}-r\boldsymbol{1})},
$$

actually delivers a higher Sharpe ratio than the $$1/K$$ benchmark. For reference: $$ \boldsymbol{\pi}_{\text{maxshp}} \approx (0.34, 0.14, 0.14, 0.33, 0.05)$$.

To analyse these questions, consider the following experimental set-up: Using simulated data we will run 
five [self-financing portfolios](https://en.wikipedia.org/wiki/Self-financing_portfolio) for 10 years, each starting with a million dollar notational. The five strategies under scrutiny are:
(i) The $$1/K$$ portfolio, (ii) $$\boldsymbol{\pi}_{\text{minvar}}$$ with known parameters, (iii) $$\boldsymbol{\pi}_{\text{minvar}}$$ with unknown parameters, 
(iv) $$\boldsymbol{\pi}_{\text{maxshp}}$$ with known parameters, and (v) $$\boldsymbol{\pi}_{\text{maxshp}}$$ with unknown parameters.
All estimated quantities will be based on at least ten years' worth of history and will be updated weekly. 
At the end all portfolios will have their variance and Sharpe ratio computed. As usual, we'll repeat
this entire process 1000 times.  

For a given matrix of simulated returns, the following function computes the wealth paths of the five strategies:

```python
π_eql = np.ones(M)/M #equally weighted portfolio
π_var = (Σinv@I)/(I@Σinv@I) #minimum variance portfolio
π_shp = (Σinv@(μ-r))/(I@Σinv@(μ-r)) #optimal sharpe portfolio

def wealth_path(dfr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    col = ['W_equal', 'W_minvar', 'W_minvar_est', 'W_sharpe', 'W_sharpe_est',]
    t_test = np.arange(int(N/2),N+1)
    dfw = pd.DataFrame(np.nan, index=t_test,columns=col)
    dfw.loc[t_test[0],:] = W_0
    pi_shp_dic = {}
    pi_var_dic = {}

    μ_hat_vec = dfr.expanding().mean()/dt
    Σ_hat_vec = dfr.expanding().cov()/dt

    for ti, t in enumerate(t_test):
        if ti%7 == 0:
            μ_hat = μ_hat_vec.loc[t].values
            Σ_hat = Σ_hat_vec.loc[t].values
            Σinv_hat = np.linalg.inv(Σ_hat)
            pi_shp_dic[t] = (Σinv_hat@(μ_hat-r))/(I@Σinv_hat@(μ_hat-r))
            pi_var_dic[t] = (Σinv_hat@I)/(I@Σinv_hat@I)
        else:
            pi_shp_dic[t] = pi_shp_dic[t-1]
            pi_var_dic[t] = pi_var_dic[t-1]

        if ti == 0:
            pass
        else:
            col_map = {
                'W_equal': π_eql,
                'W_minvar': π_var,
                'W_minvar_est': pi_var_dic[t-1],
                'W_sharpe': π_shp,
                'W_sharpe_est': pi_shp_dic[t-1],
            }
            for j in col:
                dfw.at[t,j] = dfw.at[t-1,j]*(1 + col_map[j]@dfr.loc[t])

    pi_shp = pd.DataFrame.from_dict(pi_shp_dic, orient='index')
    pi_var = pd.DataFrame.from_dict(pi_var_dic, orient='index')
    return dfw, pi_shp, pi_var  
```

Here's one such example, which in itself won't tell you much:

{% include figure.html path="assets/img/wealth.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

However, once we repeat this enough times a pattern starts to emerge. Specifically, consider the distributive 
properties of the optimal portfolios *less* the $$1/K$$ benchmark: 

<table border="1" class="dataframe" width="100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th> $$\sigma(\boldsymbol{\pi}_{\text{minvar}})-\sigma(\boldsymbol{\pi}_{1/K} )$$ </th>
      <th> $$\sigma(\hat{\boldsymbol{\pi}}_{\text{minvar}})-\sigma(\boldsymbol{\pi}_{1/K} )$$ </th>
      <th> $$Shp(\boldsymbol{\pi}_{\text{maxshp}})-Shp(\boldsymbol{\pi}_{1/K} )$$ </th>
      <th> $$Shp(\hat{\boldsymbol{\pi}}_{\text{maxshp}})-Shp(\boldsymbol{\pi}_{1/K} )$$ </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.015</td>
      <td>-0.015</td>
      <td>0.104</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.120</td>
      <td>0.161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.018</td>
      <td>-0.018</td>
      <td>-0.299</td>
      <td>-0.790</td>
    </tr>
    <tr>
      <th>2.5%</th>
      <td>-0.017</td>
      <td>-0.017</td>
      <td>-0.119</td>
      <td>-0.308</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.015</td>
      <td>-0.015</td>
      <td>0.105</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>97.5%</th>
      <td>-0.013</td>
      <td>-0.013</td>
      <td>0.342</td>
      <td>0.352</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-0.013</td>
      <td>-0.013</td>
      <td>0.503</td>
      <td>0.641</td>
    </tr>
    <tr>
      <th># &lt; 0</th>
      <td>1000</td>
      <td>1000</td>
      <td>200</td>
      <td>493</td>
    </tr>
  </tbody>
</table>
<br>

The take-away here is that the minimum variance portfolio does, in fact, deliver a lower volatility than the 
equal allocation portfolio - regardless of whether $$\boldsymbol{\Sigma}$$ is estimated or not.
As for the maximum Sharpe ratio portfolio the situation is more problematic: while 80% of the portfolios with known
parameters *do* outperform the benchmark, there's little to suggest optimisation actually helps once we introduce 
estimation into the picture. Again, the culprit is the uncertainty around the $$\hat{\boldsymbol{\mu}}$$ estimator.

In summary, even under highly idealised circumstances optimal trading strategies may not deliver as expected.
Indeed, I would strongly caution against embracing small improvements in portfolio performance from some 
complicated optimisation process vs. keeping things simple. Markowitz's $$1/K$$ portfolio allocation is a lot less irrational
than it initially appears.











[^1]: I.e. the solution to $$\min_{\boldsymbol{\pi}} \boldsymbol{\pi}^\intercal \boldsymbol{\Sigma} \boldsymbol{\pi} $$ where $$\boldsymbol{\pi}^\intercal \boldsymbol{1}=1$$. 
[^2]: I.e. the solution to $$\max_{\boldsymbol{\pi}} (\boldsymbol{\pi}^\intercal\mu - r)/\sqrt{\boldsymbol{\pi}^\intercal \boldsymbol{\Sigma} \boldsymbol{\pi}}$$ where $$\boldsymbol{\pi}^\intercal \boldsymbol{1}=1$$.