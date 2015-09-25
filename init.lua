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
gp = require 'gp.env' -- leaks a global called gp (for classing purposes)

--------------------------------
--            Standalone Modules
--------------------------------
gp.utils = require 'gp.utils'
--------------------------------
--                 Class Modules
--------------------------------
require 'gp.models'
require 'gp.kernels'
require 'gp.means'
require 'gp.grids'
require 'gp.scores'
require 'gp.samplers'
require 'gp.scientists'

return gp
