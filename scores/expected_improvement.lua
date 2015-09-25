------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Expected Improvement (EI) acquisition function:

  EI(fval; fmin) := max(0, E[fmin - fval])

The (optional) 'tradeoff' parameter helps govern
the balance between exploration and exploitation.

Authored: 2015-09-16 (jwilson)
Modified: 2015-09-22
--]]

---------------- External Dependencies
local utils = require 'gp.utils'
local math  = require('math')
local gp = require 'gp.env'
------------------------------------------------
--                          expected_improvement
------------------------------------------------
local EI, parent = torch.class('gp.scores.expected_improvement', 'gp.scores.metascore')

function EI:__init(config)
  parent.__init(self)
  local config = config or {}
  config['tradeoff']   = config.tradeoff or 0.0
  config['nFantasies'] = config.nFantasies or 100
  self.config          = config
end

function EI:__call__(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  local hyp    = hyp or model.hyp
  local config = config or self.config
  local ei = EI.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  collectgarbage()
  return ei
end

function EI.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  local nObs, nHid = X_obs:size(1), X_hid:size(1)
  if (X_obs:dim() == 1) then X_obs:resize(1, nObs); nObs=1 end
  if (X_hid:dim() == 1) then X_hid:resize(1, nHid); nHid=1 end

  ---------------- Fantasize outcomes for pending jobs:
  if torch.isTensor(X_pend) and X_pend:size(1) > 0 then
    local nPend = X_pend:size(1)
    if (X_pend:dim() == 1) then X_pend:resize(1 ,nPend); nPend=1 end
    local nOP   = nObs + nPend

    -------- Generate fantasies and append to _obs tensors
    local X_obs, Y_obs, Y_pend = X_obs, Y_obs, nilz
    Y_pend = model:fantasize(config.nFantasies, X_obs, Y_obs, X_pend, hyp)
    X_obs  = X_obs:cat(X_pend, 1)
    Y_obs  = utils.vect(Y_obs):repeatTensor(1, config.nFantasies):cat(Y_pend, 1)
  end

  -------- Compute predictive posterior at X_hid
  local pred  = model:predict(X_obs, Y_obs, X_hid, hyp, {mean=true, var=true})
  local fmins = Y_obs:min(1)

  return EI.compute(pred.mean, pred.var, fmins, config.tradeoff)
end

function EI.compute(fval, fvar, fmin, tradeoff)
  local tradeoff = tradeoff or 0.0
  local sigma, resid, zvals, ei

  if (fval:dim() > 1 and fval:size(2) > 1) then
    sigma = fvar:sqrt()
    resid = fval:mul(-1):add(utils.as_val(fmin) - tradeoff)
    zvals = resid:cdiv(sigma)
    ei    = resid:mul(utils.standard_cdf(zvals)) + sigma:mul(utils.standard_pdf(zvals))
    ei:clamp(0, math.huge):mean(2)
  else
    sigma = fvar:sqrt()
    resid = fval:mul(-1.0):add(utils.as_val(fmin) - tradeoff)
    zval  = resid:cdiv(sigma)
    ei    = resid:cmul(utils.standard_cdf(zval)) + sigma:cmul(utils.standard_pdf(zval))
    ei:clamp(0, math.huge)
  end
  return ei
end

return EI
