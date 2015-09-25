------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for gpTorch7 kernel
covariance functions.

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
require('math')

------------------------------------------------
--                                    metakernel
------------------------------------------------
local metakernel = torch.class('kernels.metakernel')

function metakernel:__init()
end

function metakernel:__call__(hyp, X, Z)
  return self.cov(hyp, X, Z)
end

function metakernel.num_hypers(X, Y)
  print('Error: num_hyp() method not implemented')
end

function metakernel.make_hypers(X, Y)
  print('Error: make_hypers() method not implemented')
end

function metakernel.feasible_hypers(hyp)
  if not hyp then return true end
  return hyp:gt(-math.huge):all() and hyp:lt(math.huge):all()
end

function metakernel.cov(hyp, X, Z)
  print('Error: cov() method not implemented')
end

function metakernel.derivatives(hyp, X, Z, gram)
  print('Error: derivatives() method not implemented')
end

function metakernel.gradients(hyp, X, Y, gram, invK)
  print('Error: gradients() method not implemented')
end

function metakernel.hyperprior(hyp)
  -------- No prior
  return 0.0
end

function metakernel.grad_hyperprior(hyp)
   -------- No prior
  return torch.zeros(hyp:nElement(),1)
end

function metakernel:__tostring__()
  return torch.type(self)
end
