# A Review of Sensitivity Methods for Differential Equations 

This respository contains all the text, code and figures used for the review paper about sentitivity methods for differential equations. This topic received different names in different communities, but the core problem is very simple and important. Given a system of differential equations 
$$
\frac{du}{dt} = f(u, \theta, t),
$$
with $u \in \mathbb R^n$ the unknow solution and $\theta \in \mathbb R^p$ a vector of parameters, how do we compute the gradient of a loss function 
$$
\mathcal L (\theta) = L ( u(\cdot, \theta) )
$$
with respect to the parameters $\theta$ of the dynamical model? 

## Open Science from Scratch: contribute to the project! 

This review started with some of the authors willing to understand this tools in a comprehensive way and gathering references from fields like statistics, applied mathematics and computer science. Unhappy with the lack of a general compendium of the different methods that exists to address this problem, we had decided to make a single document where all the methods can coexists under common ground and can be compareded under their different scopes and domains of applications.  
