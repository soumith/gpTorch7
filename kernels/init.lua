------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 kernels class.

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-18
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
local gp = require 'gp.env'
gp.kernels = {}
include('metakernel.lua')
include('ardse.lua')
include('GaussianNoise_iso.lua')

return gp.kernels
