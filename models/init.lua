------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 models class.

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-24
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------

local gp = require 'gp.env'
gp.models = {}
include('metamodel.lua')
include('gp_regressor.lua')

return gp.models
