------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initializer for gpTorch7 package.

There's probably a better way to do this...

To Do:
  - Add rockspec etc.
  - Neural net demo

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
require('paths')

------------------------------------------------
--                                   Initializer
------------------------------------------------
gpTorch7 = gpTorch7 or {}

--------------------------------
--            Standalone Modules
--------------------------------
gpTorch7['utils'] = gpTorch7.utils or require('gpTorch7_utils.lua')


--------------------------------
--                 Class Modules
--------------------------------
if not gpTorch7.models then
  gpTorch7['models'] = require('models/init.lua')
  models = nil
end

if not gpTorch7.kernels then
  gpTorch7['kernels'] = require('kernels/init.lua')
  kernels = nil
end

if not gpTorch7.means then
  gpTorch7['means'] = require('means/init.lua')
  means = nil
end

if not gpTorch7.grids then
  gpTorch7['grids'] = require('grids/init.lua')
  grids = nil
end

if not gpTorch7.scores then
  gpTorch7['scores'] = require('scores/init.lua')
  scores = nil
end

if not gpTorch7.samplers then
  gpTorch7['samplers'] = require('samplers/init.lua')
  samplers = nil
end

if not gpTorch7.scientists then
  gpTorch7['scientists'] = require('scientists/init.lua')
  scientists = nil
end

return gpTorch7