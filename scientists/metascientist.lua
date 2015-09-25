------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for gpTorch7 scientists.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-18
--]]

local gp = require 'gp.env'
------------------------------------------------
--                                 metascientist
------------------------------------------------
local scientist = torch.class('gp.scientists.metascientist')

function scientist:__init()
end

function scientist:__call__()
end

function scientist.eval()
  print('Error: eval() method not implemented')
end

function scientist.nominate()
  print('Error: nominate() method not implemented')
end

return scientist
