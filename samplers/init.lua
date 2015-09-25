------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 samplers class.

Authored: 2015-09-16 (jwilson)
Modified: 2015-09-16
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
local gp = require 'gp.env'
gp.samplers = {}
include('metasampler.lua')
include('slice.lua')

return gp.samplers
