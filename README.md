# Introduction

This is an __example__ of generating head CT perfusion map from series of CT scans by cSVD-based deconvolution in Python.

**THE ALGORITHM HAS NOT BEEN VALIDATED AND IS NOT INTENDED FOR CLINICAL USAGE**

Refer to tutorial.ipynb for more.

# References

cSVD deconvolution algorithm was implemented based on these works:

- https://github.com/SethLirette/CTP 
- https://github.com/marcocastellaro/dsc-mri-toolbox
- Zanderigo F, Bertoldo A, Pillonetto G, Cobelli Ast C. Nonlinear stochastic regularization to characterize tissue residue function in bolus-tracking MRI: assessment and comparison with SVD, block-circulant SVD, and Tikhonov. IEEE Trans Biomed Eng. 2009 May;56(5):1287-97. doi: 10.1109/TBME.2009.2013820. Epub 2009 Feb 2. PMID: 19188118.