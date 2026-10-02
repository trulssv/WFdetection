# research/

Mathematical background and methods.

* [background.md](background.md): the main document, written like the
  background/methods section of a paper. It derives every component of the
  code from first principles, with definitions, theorems and proofs:
  * §2: wavefront sets; the contact structure of $\Omega\times S^1$; why the
    wavefront set of a cartoon image is a union of Legendrian curves
    (Cor. 2.9).
  * §3: wavefront sets as 1-currents; the energy
    $E=\mathbf M_g+\nu\mathbf M(\partial\cdot)$; proofs that it has the five
    desired properties (Thm 3.8), including a Fenchel-type lower bound per
    component (Thm 3.9).
  * §4: the ray transform as an FIO; the lifted canonical relation is a
    strict contactomorphism, $C^*(ds-t\,d\varphi)=\alpha$ (Prop. 4.4), which
    links the prior to the data; the fan beam (Prop. 4.7); limited data.
  * §5: the measurement model: the fibre response (Prop. 5.3), the
    tangent-line ambiguity, and the forward-model data term (Def. 5.4).
  * §6: discretization: antipodal symmetry, blurred currents and why the
    normal component $c$ is needed (Prop. 6.3), properness of the discrete
    prior.
  * §7: MAP estimation; unsupervised training as amortized MAP (Prop. 7.2);
    the LPD architecture and the geometric constraints on it.
  * §8–10: related work, open problems, references.

Code docstrings cite results by their number in this document.
