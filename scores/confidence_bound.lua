------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Confidence bound acquisition functions:
  UCB(fval, fvar) := fval + sqrt(fvar) [upper]
  LCB(fval, fvar) := fval - sqrt(fvar) [lower]

By default, the present file computes the
negative LCB (a quantity we seek to maximize).

The (optional) 'tradeoff' parameter helps govern
the balance between exploration and exploitation.

Authored: 2015-09-17 (jwilson)
Modified: 2015-09-22
--]]

---------------- External Dependencies
local utils = require 'gp.utils'
local gp = require 'gp.env'
------------------------------------------------
--                              confidence_bound
------------------------------------------------
local conf_bound, parent = torch.class('gp.scores.confidence_bound', 
				       'gp.scores.metascore')

function conf_bound:__init(config)
  local config = config or {}
  config['tradeoff']   = config.tradeoff or 1.0
  config['nFantasies'] = config.nFantasies or 100
  config['bound']      = config.bound or 'lower'
  config['sign']       = config.sign  or -1.0
  self.config          = config
end

function conf_bound:__call__(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  local hyp    = hyp or model.hyp
  local config = config or self.config
  return conf_bound.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
end

function conf_bound.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
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

  -------- Compute requested (signed) confidence bound 
  return conf_bound.compute(pred.mean, pred.var, config)
end

function conf_bound.compute(fval, fvar, config)
  local tradeoff = config.tradeoff or 1.0
  local bound    = config.bound:lower() or 'lower'
  local sign     = config.sign or -1.0

  -------- Compute requested confidence bound 
  local val
  if bound == 'lower' then
    val = conf_bound.LCB(fval, fvar, tradeoff)
  elseif bound == 'upper' then
    val = conf_bound.UCB(fval, fvar, tradeoff)
  end

  -------- Return signed value
  if sign > 0.0 then
    return val
  else
    return -val
  end
end

function conf_bound.UCB(fval, fvar, tradeoff)
  local tradeoff = tradeoff or 1.0
  local ucb
  if (fval:dim() > 1 and fval:size(2) > 1) then
    ucb = torch.mean(fval:add(fvar:sqrt():mul(tradeoff)), 2)
  else
    ucb = fval:add(fvar:sqrt():mul(tradeoff))
  end
  return ucb
end

function conf_bound.LCB(fval, fvar, tradeoff)
  local tradeoff = tradeoff or 1.0
  local lcb
  if (fval:dim() > 1 and fval:size(2) > 1) then
    lcb = torch.mean(fval:add(-fvar:sqrt():mul(tradeoff)), 2)
  else
    lcb = fval:add(-fvar:sqrt():mul(tradeoff))
  end
  return lcb
end

return conf_bound
