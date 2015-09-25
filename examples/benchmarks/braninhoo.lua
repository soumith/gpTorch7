------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Branin-Hoo 2D benchmarking function:

  f(x1, x2) = (z2 + c1*z1^2 + c2*z1 - 6.0)^2
               + c3*cos(z1) + 10.0

where z1 := 15*x1 - 5, z2 := 15*x2.

Global Minima:
(x1, x2) ~= {(0.124, 0.818), (0.543, 0.152), (0.962, 0.165)}
(z1, z2) ~= {(-3.142, 12.275), (3.142, 2.275), (9.425, 2.475)} 
  
Authored: 2015-09-18 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local c1 = -5.1/(4.0*math.pi*math.pi)
local c2 = 5.0/math.pi
local c3 = 10.0 - 10.0/(8.0*math.pi)
------------------------------------------------
--                                     braninhoo
------------------------------------------------

function braninhoo(X)
  if (X:dim() == 1 or X:size(1) == X:nElement()) then
    X = X:resize(1, X:nElement())
  end
  assert(X:size(2) == 2)

  -------- Transform X -> Z
  local Z1 = X:select(2,1):clone():mul(15.0):add(-5.0)
  local Z2 = X:select(2,2):clone():mul(15.0)

  -------- Compute Branin-Hoo function
  local Y  = Z2:add(torch.pow(Z1, 2):mul(c1)):add(torch.mul(Z1, c2)):add(-6.0):pow(2)
               :add(Z1:clone():cos():mul(c3):add(10.0))
  return Y
end

