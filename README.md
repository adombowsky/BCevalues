# Introduction
For any test statistic $T(X)$, a Besag-Clifford e-value is
\begin{equation*}
\hat E_M(X) = \frac{(M+1)T(X)}{T(X) + \sum_{m=1}^M T(Y^{(m)}) } 
\end{equation*}
where $(X,Y^{(1)}, \dots, Y^{(M)})$ are sampled using the parallel algorithm of Besag and Clifford.
This repository provides code to implement the experiments and applications in ``Besag-Clifford e-values for unnormalized testing."

## AR(1) Process
We test $H_0: \mathcal N(0,1)$ against $H_1: \mathcal N(\mu,1)$. The BC e-value is
\begin{equation*}
\hat E_M(X) = \frac{(M+1)\exp(\mu x)}{\exp(\mu x) + \sum_{m=1}^M \exp(\mu Y^{(m)}) }.
\end{equation*}


## Sequential PoE
The null hypothesis is a product of experts (PoE) with $2$ experts and the alternative is the $\mathcal N(0,1)$ distribution.
The BC e-value is constructed from
\begin{equation*}
    T(x_i) = \frac{e^{-\frac{(x_i-\mu)^2}{2}}}{\sqrt{2\pi \sigma^2}} \prod_{w=1}^2 \left( 1 + \frac{(x_i-\psi_w)^2}{\theta_w} \right)^{\frac{\theta_w+1}{2}}.
\end{equation*}

## Sequential Composite Alternatives
The null is $\mathcal N(0,1)$ and the alternative is $\mathcal N(\mu, \sigma^2)$ for unknown $\mu$ and $\sigma^2$.
The BC e-value is constructed from
\begin{equation*}
 T(x_t \mid x_{1:(t-1)}) = \frac{\N(x_t; \bar x_t, \hat \sigma_t^2)}{\mathcal N(x_t; 0, 1)}.
\end{equation*}