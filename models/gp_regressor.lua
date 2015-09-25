------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
To Do:
  - Debug hyperpriors

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-25
--]]

---------------- External Dependencies
torch = require('torch')
optim = require('optim')
math  = require('math')

local utils = gpTorch7.utils

---------------- Constants
configs            = {}
configs['slice']   = {}
configs['lsearch'] = {verbose    = true,
                      maxIter    = 100}

configs['lbfgs']   = {verbose    = true,
                      maxIter    = 1e3,
                      maxEval    = 1e3,
                      lineSearch = optim.lswofle,
                      lineSearchOptions = configs.lsearch}

------------------------------------------------
--                    Gaussian Process Regressor
------------------------------------------------
local GPR, parent = torch.class('models.gp_regressor', 'models.metamodel')

--------------------------------
--                Initialization
--------------------------------
function GPR:__init(config, kernel, nzModel, mean, cache, X, Y)
  parent.__init(self)
  local cache  = cache or {}
  self.config  = cache.config or config or {}
  self.kernel  = cache.kernel or gpTorch7.kernels[kernel or config.kernel]()
  self.mean    = cache.mean or gpTorch7.means[mean or config.mean]()
  self.hyp     = cache.hyp or nil

  -------- Optional observation noise model
  if not config.noiseless then
    self.nzModel = cache.nzModel or gpTorch7.kernels[nzModel or config.nzModel]()
  end
  self:init(X, Y)
end

function GPR:init(X, Y)
  -------- Short-circuit
  if not (X and Y) then return end

  -------- Local Variables
  local config = self.config

  -------- Update Config
  config['xDim'] = config.xDim or X:size(2)
  config['yDim'] = config.xDim or Y:size(2)

  -------- Establish hyperparameter info
  config['nHypers']      = {}
  config.nHypers['cov']  = self.kernel.num_hypers(X, Y)
  config.nHypers['mean'] = self.mean.num_hypers(X, Y)
  if not config.noiseless then
    config.nHypers['nz'] = self.nzModel.num_hypers(X, Y)
  end
  
  config['nHyp'] = 0
  for key,val in pairs(config.nHypers) do
    if key ~= 'mean' then
      config.nHyp = config.nHyp + val
    end
  end

  -------- To do: 
  -- config['ranges'] = config.ranges or {}
  -- config.ranges['hyp']   = torch.Tensor(config.nHyp, 2)
  -- for idx = 1,config.nHyp do
  --   -- config.ranges.hyp[idx][1] = self.hyp[idx].min
  --   -- config.ranges.hyp[idx][2] = self.hyp[idx].max
  -- end

  -------- Generate hyperparameters
  self.mean_hyp = self.mean.make_hypers(X, Y) -- temp hack
  self.hyp = self:parse_hypers(self:make_hypers(X, Y, true))

  -------- Establish slice sampler widths
  configs.slice['widths'] = torch.Tensor(1,self.config.nHypers.cov):fill(0.1)
  if not config.noiseless then
    configs.slice.widths = configs.slice.widths:cat(torch.Tensor{{100}}, 2)
  end

  return self
end

--------------------------------
--                    GP Methods
--------------------------------
function GPR:cov_func(X, Z, hyp, noiseless, cov)
  local hyp = self:parse_hypers(hyp or self.hyp)

  -------- Compute covariance matrix
  local cov = cov or self.kernel(hyp.cov, X, Z)

  -------- Apply noise model? 
  if not (Z or noiseless or self.config.noiseless) then
    cov:add(self.nzModel(hyp.nz, X))
  end
  return cov
end

function GPR:mean_func(X, hyp)
  local hyp = self:parse_hypers(hyp or self.hyp)
  return self.mean(hyp.mean, X)
end

function GPR:resid_func(X, Y, hyp)
  return utils.vect(Y - self:mean_func(X, hyp), 'F', true)
end

function GPR:loglik(X, Y, hyp, cache, w_priors)
  -------- Parse Hyperparameters
  local hyp   = self:parse_hypers(hyp or self.hyp)
  local noisy = not self.config.noiseless

  -------- Compute required terms
  local cache = cache or {}
  local resid = cache.resid or self:resid_func(X, Y, hyp)
  local chol  = cache.chol or torch.potrf(cache.cov or self:cov_func(X, nil, hyp), 'L')
  local alpha = cache.alpha or torch.potrs(resid, chol, 'L')

  -------- Compute data log-likelihood
  local llh = -torch.sum(torch.log(torch.diag(chol))) - 0.5*torch.dot(resid, alpha)

  -------- Apply hyperpriors
  if w_priors ~= false then
    llh = llh + self.kernel.hyperprior(hyp.cov)
    if noisy then
      llh = llh + self.nzModel.hyperprior(hyp.nz)
    end
  end

  collectgarbage()
  return llh
end

function GPR:predict(X, Y, Z, hyp, req, cache)
  local hyp   = hyp or self.hyp
  local req   = req or {mean = true, var = true}
  local cache = cache or {}
  local K_xx  = cache.cov or self:cov_func(X, nil, hyp)
  local chol  = cache.chol or torch.potrf(K_xx, 'L')
  local K_zx  = cache.cross_cov or self:cov_func(Z, X, hyp)
  local res   = {} -- result table

  -------- Posterior mean
  -- mu(f(z)|x,y,z) := mu(z) + K_zx*K_xx^(-1)(f(x) - mu(x))
  if req.mean then
    res['mean'] = utils.vect(self:mean_func(Z, hyp):add(torch.mm(K_zx,
          torch.potrs(cache.resid or self:resid_func(X, Y, hyp), chol))))
  end

  -------- Posterior variance/covariance
  -- cov(f(z)|x,f(x),z) := K_zz - K_zx*K_xx^(-1)*K_xz
  if req.var or req.cov then
    local beta = torch.trtrs(K_zx:t(), chol, 'L')

    if req.var then
      -- Note: diag(K_zz) = amp2*(exp(0) + eps), where eps is a stabilizing factor
      if not self.config.noiseless then
        res['var'] = beta:pow(2):sum(1):mul(-1.0):add(utils.as_val(hyp.cov[1])*(1+1e-6))
                     :add(utils.as_val(hyp.nz[1]))
      else
        res['var'] = beta:pow(2):sum(1):mul(-1.0):add(utils.as_val(hyp.cov[1])*(1+1e-6))
      end
    end

    if req.cov then
      res['cov'] = self:cov_func(Z, nil, hyp):add(-torch.mm(beta:t(), beta))
    end
  end

  collectgarbage()
  return res
end

function GPR:fantasize(nFantasies, X, Y, Z, hyp, cache)
  local cache = cache or {}

  ---- Posterior mean and covariance
  local post = self:predict(X, Y, Z, hyp, {mean=true, cov=true}, cache)
  
  ------- Fantasize (i.e. sample from predictive posterior)
  local chol    = torch.potrf(post.cov)
  local fantasy = post.mu:repeatTensor(1, nFantasies):add(torch.mm(
                      chol, torch.randn(Z:size(1), nFantasies)))
  collectgarbage()
  return fantasy
end

--------------------------------
--               Hyperparameters
--------------------------------
function GPR:make_hypers(X, Y, as_tensor)
  local as_tensor = as_tensor or false
  local hyp = self.kernel.make_hypers(X, Y)

  if not self.config.noiseless then
    hyp = torch.cat(hyp, self.nzModel.make_hypers(X, Y), 1)
  end

  if as_tensor then
    return hyp
  else
    return self:parse_hypers(hyp, false, 'mean')
  end
end

function GPR:parse_hypers(hyp, as_tensor, exclude)
  local config = self.config
  local nHyp
  if torch.isTensor(hyp) then
    nHyp = hyp:nElement()
  else
    nHyp = config.nHyp
  end

  -------- Convert hyp table to tensor
  if type(hyp) == 'table' and utils.tbl_size(hyp) == nHyp then
    hyp = torch.Tensor(hyp):resize(nHyp, 1)
  end

  -------- Divide hyp tensor into sub-tensors (per module)
  if torch.isTensor(hyp) then
    local hyp_tbl  = {}
    hyp_tbl['cov'] = hyp:sub(1, config.nHypers.cov):clone()
    if not self.config.noiseless then
      hyp_tbl['nz'] = hyp:sub(config.nHypers.cov+1, config.nHyp):clone() -- temp hack
    end
    hyp = hyp_tbl
  end

  -------- Add in mean function hyperparameters (temp hack)
  if type(hyp) == 'table' and not hyp.mean then
    hyp['mean'] = self.mean_hyp:clone()
  end

  -------- Exclude subtensor(s) from hyperparameter table
  if type(exclude) == 'string' then
    hyp[exclude] = nil
  elseif  type(exclude) == 'table' then
    local key 
    for k in 1,utils.tbl_size(exclude) do
      hyp[key] = nil
    end
  end

  -------- Convert hyperparameter table to tensor
  if as_tensor then
    local hyp_tensor = torch.Tensor(nHyp, 1) -- +1 for mean (temp hack)
    local N, idx = 0, 1
    for key, subtensor in pairs(hyp) do
      N = subtensor:nElement()
      hyp_tensor:sub(idx, idx+N-1):copy(subtensor)
      idx = idx + N
    end
    hyp = hyp_tensor
  end
  return hyp
end

function GPR:feasible_hypers(hyp)
  local hyp = self:parse_hypers(hyp)
  local feasible = self.mean.feasible_hypers(hyp.mean)  and
                   self.kernel.feasible_hypers(hyp.cov)

  if feasible and not self.config.noiseless then
     feasible = feasible and self.nzModel.feasible_hypers(hyp.nz)
  end

  return feasible
end

function GPR:grad_hypers(X, Y, hyp, cache, noiseless)
  -- To Do: Add in mean function hypers
  local hyp   = hyp or self.hyp
  local cache = cache or {}
  local cache = cache or {}
  local resid = cache.resid or self:resid_func(X, Y, hyp)
  local grads = self.kernel.gradients(hyp.cov, X, resid, cache)

  if not noiseless then
    grads = grads:cat(self.nzModel.gradients(hyp.nz, X, resid, cache), 1)
  end

  collectgarbage()
  return grads
end

function GPR:protocol(f, X, Y, f_args, hyp, w_priors)
  function closure(log_hyp)
    local N     = X:size(1)
    local hyp   = self:parse_hypers(torch.exp(log_hyp))
    local noisy = not self.config.noiseless

    local cache    = {}
    cache['resid'] = self:resid_func(X, Y, hyp)
    cache['cov']   = self.kernel(hyp.cov, X)
    if noisy then
      cache['cov_nz'] = cache['cov'] + self.nzModel(hyp.nz, X)
      cache['chol'] = torch.potrf(cache.cov_nz)
      cache['invK'] = torch.inverse(cache.cov_nz)
    else   
      cache['chol'] = torch.potrf(cache.cov)
      cache['invK'] = torch.inverse(cache.cov)
    end
    cache['alpha'] = torch.mm(cache.invK, cache.resid)

    local nll = -self:loglik(X, Y, hyp, cache, w_priors)

    cache['shared'] = torch.mm(cache.alpha, cache.alpha:t()):add(-cache.invK):t():resize(N, N, 1) -- better name?
    local ngrad     = -self:grad_hypers(X, Y, hyp, cache)

    collectgarbage()
    return nll, ngrad
  end

  local log_hyp
  if torch.isTensor(hyp) then
    log_hyp = torch.log(hyp)
  else
    log_hyp = self:make_hypers(X, Y, true):log()
  end
  -------- Update mean hyperparameters (hack!)
  self.mean_hyp = self.mean.make_hypers(X, Y)
  return f(closure, log_hyp, f_args)
end

function GPR:checkgrad(X, Y)
  return self:protocol(optim.checkgrad, X, Y, nil, nil, false)
end

function GPR:optimize_hypers(X, Y, config)
  local config = config or configs.lbfgs
  return self:protocol(optim.lbfgs, X, Y, config)
end

-------- Not a protocol method because samplers use a different closure 
function GPR:sample_hypers(X, Y, X0, config, as_table)
  function closure(hyp)
    local hyp = torch.exp(hyp):resize(hyp:nElement(), 1)
    if self:feasible_hypers(hyp) then
      return self:loglik(X, Y, hyp)
    else
      return -math.huge
    end
  end

  local X0 = X0 or self:make_hypers(X, Y, true):log()
        X0:resize(1, X0:nElement())

  -------- Update mean hyperparameters (hack!)
  self.mean_hyp = self.mean.make_hypers(X, Y)

  local config  = config or configs.slice
  if not config.widths then
    config['widths'] = configs.slice.widths
  end

  local samples = gpTorch7.samplers.slice()(closure, X0, config)

  if as_table then -- temp hack
    samples = self:parse_hypers(samples:t(), false, nil, X)
  end

  return samples
end
