# NAMT theory

## Event geometry

For muon event $i$, let

$$
\mathbf h_{ik}=(x_{ik},y_{ik})^{\mathsf T}
$$

denote the transverse hit at detector plane $k$ with axial coordinate $z_k$. The first two hits define the incident trajectory

$$
\mathbf r_{i,\perp}(z)
=\mathbf h_{i1}
+\frac{z-z_1}{z_2-z_1}
(\mathbf h_{i2}-\mathbf h_{i1}),
\qquad
\mathbf r_i(z)
=\bigl(\mathbf r_{i,\perp}(z)^{\mathsf T},z\bigr)^{\mathsf T}.
$$

At a downstream plane $k$, the predicted hit and residual are

$$
\widehat{\mathbf h}_{ik}=\mathbf r_{i,\perp}(z_k),
\qquad
\boldsymbol\varepsilon_{ik}
=\mathbf h_{ik}-\widehat{\mathbf h}_{ik}.
$$

The incident slope and zenith angle satisfy

$$
\mathbf t_i
=\frac{\mathbf h_{i2}-\mathbf h_{i1}}{z_2-z_1}
=(t_{x,i},t_{y,i})^{\mathsf T},
\qquad
\sec\theta_i=\sqrt{1+\lVert\mathbf t_i\rVert_2^2}.
$$

## Target response

Let $\Omega$ be the reconstruction volume and let

$$
\lambda(\mathbf r)=\frac{1}{X_0(\mathbf r)},
$$

where $X_0(\mathbf r)$ is the local radiation length. For momentum $p$ and $\beta=v/c$, define

$$
g(p)=\left(\frac{13.6\,\mathrm{MeV}}{\beta pc}\right)^2.
$$

Conditioned on $g$, scattering along the incident trajectory contributes the position-residual variance

$$
V^{\mathrm S}_{ik}(g,\lambda)
=\int_{z_{\min}}^{z_{\max}}
g\,\sec^3\theta_i\,(z_k-z)^2
\lambda\bigl(\mathbf r_i(z)\bigr)\,\mathrm dz.
$$

For coordinate $d\in\{x,y\}$, the target-induced residual density is

$$
p^{\mathrm S}_{ikd}(s\mid g,\lambda)
=\mathcal N\bigl(s;0,V^{\mathrm S}_{ik}(g,\lambda)\bigr).
$$

## Instrument response

If each measured coordinate has independent variance $\sigma_{\mathrm{hit}}^2$, define

$$
a_k=\frac{z_k-z_1}{z_2-z_1}.
$$

Propagation of detector error gives

$$
V_k^{\mathrm{det}}
=\sigma_{\mathrm{hit}}^2
\bigl[1+(1-a_k)^2+a_k^2\bigr].
$$

Define

$$
(x_i^{(0)},y_i^{(0)})^{\mathsf T}=\mathbf r_{i,\perp}(0),
\qquad
\boldsymbol\chi_i
=\bigl(x_i^{(0)},y_i^{(0)},t_{x,i},t_{y,i}\bigr)^{\mathsf T}.
$$

Bounded corrections $\mu_{kd}(\boldsymbol\chi_i)$ and $\delta_{kd}(\boldsymbol\chi_i)$ are learned from empty-detector events by maximum likelihood. The instrument residual density is

$$
p^{\mathrm I}_{ikd}(b\mid\boldsymbol\chi_i)
=\mathcal N\left(
b;
\mu_{kd}(\boldsymbol\chi_i),
V_k^{\mathrm{det}}\exp\delta_{kd}(\boldsymbol\chi_i)
\right).
$$

Conditioned on $g$, the observed residual is the sum of the independent target and instrument contributions:

$$
\rho_{ikd}(\varepsilon\mid g,\boldsymbol\chi_i,\lambda)
=\int_{-\infty}^{\infty}
p^{\mathrm S}_{ikd}(s\mid g,\lambda)
p^{\mathrm I}_{ikd}(\varepsilon-s\mid\boldsymbol\chi_i)
\,\mathrm ds.
$$

## Momentum-marginalized likelihood

Let $G=g(P)$ have density $p_G(g)$. The momentum density used in the experiments is derived from the CAPRICE94 ground-level muon measurements reported by Kremer et al., *Physical Review Letters* 83, 4241 (1999). Collecting the downstream residuals of event $i$ as $\boldsymbol\varepsilon_i$, the event likelihood is

$$
p(\boldsymbol\varepsilon_i\mid\boldsymbol\chi_i,\lambda)
=\int_0^{\infty}
p_G(g)
\prod_{k\in\mathcal D}
\prod_{d\in\{x,y\}}
\rho_{ikd}
\bigl(\varepsilon_{ikd}\mid g,\boldsymbol\chi_i,\lambda\bigr)
\,\mathrm dg.
$$

Here $\mathcal D=\{3\}$ for NAMT-3P and $\mathcal D=\{3,4\}$ for NAMT-4P.

## Reconstruction

For $N$ independent muon events,

$$
\widehat{\lambda}
=\underset{\lambda\ge 0}{\mathrm{argmin}}
\left[
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(\boldsymbol\varepsilon_i\mid\boldsymbol\chi_i,\lambda)
+\alpha\,\mathrm{TV}(\lambda)
\right],
$$

with equal-axis L1 total variation

$$
\mathrm{TV}(\lambda)
=\int_{\Omega}
\left(
|\partial_x\lambda|
+|\partial_y\lambda|
+|\partial_z\lambda|
\right)
\,\mathrm dV.
$$

The reported image is

$$
\widehat{\Lambda}(x,y)
=\int_{z_{\min}}^{z_{\max}}
\widehat{\lambda}(x,y,z)
\,\mathrm dz.
$$
