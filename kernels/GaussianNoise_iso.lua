------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Isotropic Gaussian noise model.

Authored: 2015-09-14 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
local utils = require 'gp.utils'

------------------------------------------------
--                             GaussianNoise_iso
------------------------------------------------
local nzModel, parent = torch.class('gp.kernels.GaussianNoise_iso', 
				    'gp.kernels.metakernel')

function nzModel:__init()
  parent.__init(self)
end

function nzModel.num_hypers(X, Y)
  return 1
end

function nzModel.make_hypers(X, Y, nz_std)
  local nz_std = torch.Tensor(1,1):fill(nz_std or 1e-3)
  return nz_std   
end

function nzModel.feasible_hypers(hyp)
  if not hyp then return true end
  return not torch.lt(hyp, 0.0):any()
end

function nzModel.cov(hyp, X, Z)
  if torch.isTensor(Z) and X:size(1) ~= Z:size(1) then
    return 0
  else
    return torch.eye(X:size(1)):mul(utils.as_val(hyp))
  end
end

function nzModel.derivatives(hyp, X)
  return torch.eye(X:size(1)):mul(utils.as_val(hyp))
end

function nzModel.gradients(hyp, X, Y, cache)
  local shared, invK, alpha = cache.shared, cache.invK, cache.alpha
  if not shared then
    if not (invK or alpha) then
      if not (cache.cov or cache.cov_nz) then 
        print('Error: nzModel.gradients() requires an input covariance matrix!')
      end

      if cache.cov_nz then
        invK = invK or torch.inverse(cache.cov_nz)
      else
        invK = invK or torch.inverse(cache.cov + self.cov(hyp, X))
      end

      alpha = alpha or torch.mm(invK, Y)
    end
    shared = (torch.mm(alpha, alpha:t()) - invK):t()
  end

  -------- Compute gradients w.r.t. noise model parameter
  local grads = nzModel.derivatives(hyp, X)
        grads = grads:cmul(shared):sum(2):sum(1):mul(0.5)

  -------- Factor in hyperprior 
  grads = grads + nzModel.grad_hyperprior(hyp)
  return grads
end


------------------------------------------------
--                           Developer's Section
------------------------------------------------

---- Seems buggy
-- function nzModel.hyperprior(hyp)
--  -------- Noise variance: Horseshoe prior
--  return torch.log(torch.log(1.0 + 0.01/utils.as_val(hyp)))
-- end

-- function nzModel.grad_hyperprior(hyp)
--   -------- Noise variance: gradient of Horseshoe prior
--   local nz_var = utils.as_val(hyp)
--   local u      = 1.0 + .01/nz_var
--   local grad   = -0.01/(u*torch.log(u)*nz_var^2)

--   -------- Transform gradient to account for log
--   grad = hyp:clone():mul(2):div(grad)

--   return grad
-- end


return nzModel
