------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for gpTorch7 acquisition
functions (scores). 

Authored: 2015-09-16 (jwilson)
Modified: 2015-09-21
--]]

---------------- External Dependencies
------------------------------------------------
--                               metaacquisition
------------------------------------------------
local score = torch.class('scores.metascore')

function score:__init()
end

function score:__call__(gp, X_obs, Y_obs, X_hid, X_pend)
  return score.eval(gp, X_obs, Y_obs, X_hid, X_pend)
end

function score.eval()
  print('Error: eval() method not implemented')
end