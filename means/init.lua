------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 mean functions
class.

Authored: 2015-09-15 (jwilson)
Modified: 2015-09-15
--]]

---------------- External Dependencies
------------------------------------------------
--                                   Initializer
------------------------------------------------
local gp = require 'gp.env'
gp.means = {}
include('metamean.lua')
include('constant.lua')

return means
