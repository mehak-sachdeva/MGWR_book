
# Multiscale Geographically Weighted Regression - Poisson dependent variable


The model has been explored and tested for multiple parameters on real and simulated datasets. The research includes the following outline with separate notebooks for each part.


**Notebook Outline:**  
  
**[Introduction Notebook (current)](Poisson_MGWR.ipynb)**
- [Introduction](#Introduction)
 - [Introduction to the problem](#Introduction-to-the-project)
 - [Important Equations](#Statistical-Equations) 
- [Approaches Explored](#Approaches-Explored)
- [References](#References)

**[Initial module changes and univariate model check ](https://mehak-sachdeva.github.io/MGWR_book/Html/Poisson_MGWR_univariate_check.html)**
- [Setup with libraries](#Set-up-Cells)
- [Fundamental equations for Poisson MGWR](#Fundamental-equations-for-Poisson-MGWR)
- [Example Dataset](#Example-Dataset)
- [Helper functions](#Helper-functions)
- [Univariate example](#Univariate-example)
    - [Parameter check](#Parameter-check)
    - [Bandwidths check](#Bandwidths-check)

**[Simulated Data example](Simulated_data_example_Poisson-MGWR.ipynb)**
- [Setup with libraries](#Set-up-Cells)
- [Create Simulated Dataset](#Create-Simulated-Dataset)
    - [Forming independent variables](#Forming-independent-variables)
    - [Creating y variable with Poisson distribution](#Creating-y-variable-with-Poisson-distribution)
- [Univariate example](#Univariate-example)
    - [Bandwidth: Random initialization check](#Bandwidth:-Random-initialization-check)
    - [Parameters check](#Parameters-check)
- [Multivariate example](#Multivariate-example)
    - [Bandwidths: Random initialization check](#Bandwidths:-Random-initialization-check)
    - [Parameters check](#Parameters-check)
- [Global model parameter check](#Global-model-parameter-check)
 
**[Real Data example](Real_data_example_Poisson-MGWR.ipynb)**

- [Setup with libraries](#Set-up-Cells)
- [Tokyo Mortality Dataset](#Tokyo-Mortality-Dataset)
- [Univariate example](#Univariate-example)
    - [Bandwidth: Random initialization check](#Bandwidth:-Random-initialization-check)
    - [Parameter check](Parameter-check)
- [Multivariate example](#Multivariate-example)
    [Bandwidths: Random initialization check](#Bandwidths:-Random-initialization-check)
- [MGWR bandwidths](#MGWR-bandwidths)
- [AIC, AICc, BIC check](#AIC,-AICc,-BIC-check)

**[Monte Carlo Simulation Visualization](Poisson_MGWR_MonteCarlo_Results.ipynb)**
 
- [Setup with libraries](#Set-up-Cell)
- [List bandwidths from pickles](#List-bandwidths-from-pickles)
- [Parameter functions](#Parameter-functions)
- [GWR bandwidth](#GWR-bandwidth)
- [MGWR bandwidths](#MGWR-bandwidths)
- [AIC, AICc, BIC check](#AIC,-AICc,-BIC-check)
    - [AIC, AICc, BIC Boxplots for comparison](#AIC,-AICc,-BIC-Boxplots-for-comparison)
- [Parameter comparison from MGWR and GWR](#Parameter-comparison-from-MGWR-and-GWR)

---

# Introduction

## Introduction to the project

A recent addition to the local statistical models in PySAL is the implementation of Multiscale Geographically Weighted Regression (MGWR) model, a multiscale extension to the widely used approach for modeling process spatial heterogeneity - Geographically Weighted Regression (GWR). GWR is a local spatial multivariate statistical modeling technique embedded within the regression framework that is calibrated and estimates covariate parameters at each location using borrowed data from neighboring observations. The extent of neighboring observations used for calibration is interpreted as the indicator of scale for the spatial processes and is assumed to be constant across covariates in GWR. MGWR, using a back-fitting algorithm relaxes the assumption that all processes being modeled operate at the same spatial scale and estimates a unique indicator of scale for each process.
The GWR model in PySAL can currently estimate Gaussian, Poisson and Logistic models though the MGWR model is currently limited to only Gaussian models. This project aims to expand the MGWR model to nonlinear local spatial regression modeling techniques where the response outcomes may be discrete (following a Poisson distribution). This will enable a richer and holistic local statistical modeling framework to model multi-scale process heterogeneity for the open source community.

## Statistical Equations

A conventional Poisson regression model is written as:

\begin{align}
O_i ~ Poisson[E_i exp ({\sum} {\beta} & _k x _{k,i})] \\
\end{align}

where  $x_{k,1}$ is the kth explanatory variable in place i and the ${\beta}_ks$ are the parameters and Poisson indicates a Poisson distribution with mean $\lambda$.

Nakaya et.al. (2005) introduced the concept of allowing parameter values to vary with geographical location ($u_i$), which is a vector of two dimensional co-ordinates describing the location i. The Poisson model for geographically varying parameters can be written as:

\begin{align}
O_i ~ Poisson[E_i exp ({\sum} {\beta} & _k (u_i) x _{k,i})] \\
\end{align}

The Geographically Weighted Poisson Regression model (GWPR) is estimated using a modified local Fisher scoring procedure, a form of iteratively reweighted least squares (IRLS). In this procedure, the following matrix computation of weighted least squares should be repeated to update parameter estimates until they converge (Nakaya et.al., 2005):

\begin{align}
\beta^{(l+1)} (u_i) = (X^{t} W (u_i) A(u_i)^{(l)} X)^{-1} X^{t} W (u_i) A (u_i) ^{(l)} z (u_i){(l)} \\
\end{align}

# Approaches Explored

# References

1. Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale Geographically Weighted Regression (MGWR). Annals of the American Association of Geographers, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480


2. Nakaya, T., Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2005). Geographically weighted Poisson regression for disease association mapping. Statistics in Medicine, 24(17), 2695–2717. https://doi.org/10.1002/sim.2129


3. Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J. (2019). Inference in Multiscale Geographically Weighted Regression. Geographical Analysis, gean.12189. https://doi.org/10.1111/gean.12189

