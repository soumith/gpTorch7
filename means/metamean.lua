------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for gpTorch7 mean functions.

Authored: 2015-09-15 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
require('math')
local gp = require 'gp.env'
------------------------------------------------
--                                      metamean
------------------------------------------------
local metamean = torch.class('gp.means.metamean')

function metamean:__init()
end

function metamean.num_hypers(X, Y)
  print('Error: num_hyp() method not implemented')
end

function metamean.make_hypers(X, Y)
  print('Error: make_hypers() method not implemented')
end

function metamean.feasible_hypers(hyp)
  if not hyp then return true end
  return hyp:gt(-math.huge):all() and hyp:lt(math.huge):all()
end

function metamean.predict(hyp, X)
  print('Error: predict() method not implemented')
end

function metamean:__call__(hyp, X, Z)
  return self.predict(hyp, X, Z)
end

function metamean:__tostring__()
  return torch.type(self)
end

return metamean
