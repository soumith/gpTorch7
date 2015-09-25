[![Join the chat at https://gitter.im/torch/torch7](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/j-wilson/gpTorch7?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<a name="gpTorch7.intro.dok"/>
# gpTorch7 #

__gpTorch7__ is a framework for Gaussian Process-based machine learning implemented in Torch7. This package is currently still in an alpha-build stage; so, please feel free to pass along suggestions and/or feedback.
<a name="gpTorch7.content.dok"/>
## Package Content ##

Directory    | Content 
:-------------:|:----------------------
models   | Overarching model class (e.g. Gaussian processes)
kernels  | Covariance functions / Noise models
means    | Mean functions
scores   | Acquisition functions (incl. EI, UCB)
samplers | Sampling methods (e.g. slice sampling)
examples | Demos / Benchmark Functions
<a name="gpTorch7.dev.dok"/>
## Developers' Notes ##
Please be sure to grab an updated copy of Torch7 prior to using gpTorch7, as the package relies on newer functions (e.g. torch.potrf/s).

Further documentation coming soon.
