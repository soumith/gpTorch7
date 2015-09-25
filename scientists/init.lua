------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 scientist class.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-18
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
local gp = require 'gp.env'
gp.scientists = {}
include('metascientist.lua')
include('random_search.lua')
include('bayesopt.lua')
return gp.scientists
