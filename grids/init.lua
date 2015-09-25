------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 grid class.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-21
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
local gp = require 'gp.env'
gp.grids = {}
include('metagrid.lua')
include('random.lua')
return gp.grids
