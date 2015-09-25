------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Constant predictor mean function.

Authored: 2015-09-15 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
require('math')

------------------------------------------------
--                                      Constant
------------------------------------------------
local const, parent = torch.class('means.constant', 'means.metamean')

function const:__init()
  parent.__init(self)
end

function const.num_hypers(X, Y)
  return Y:size(2)
end

function const.make_hypers(X, Y)
  return Y:mean(1)
end

function const.predict(hyp, X)
  local hyp  = hyp
  local nHyp = hyp:nElement()
  if hyp:dim() == 1 or hyp:size(1) == nHyp then
    hyp = hyp:clone():resize(1, nHyp)
  end
  return hyp:repeatTensor(X:size(1), 1)
end

---- To do:
-- function const.gradients(hyp, X, resid)
--   return resid:mean(1) -- wrongo
-- end


