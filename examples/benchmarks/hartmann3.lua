------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Hartmann 3D benchmarking function:

  f(x) = -a * exp(-sum(A o (kron(x, [1 1 1 1]') - P)^2, 2))

where:
  A := |3.0 10 30|
       |0.1 10 35|
       |3.0 10 30|
       |0.1 10 35|

  P := |.3689 .1170 .2673|
       |.4699 .4387 .7470|
       |.1091 .8732 .5547|
       |.0381 .5743 .8828| 
  
  a := |1.0, 1.2, 3.0, 3.2|

and, domain(x) := [0,1]^3

Global Minima:
f(x*) = -3.862878 at x* := (0.114614, 0.555649, 0.852547)

  
Authored: 2015-09-18 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local A = -torch.Tensor{{3.0, 10, 30},
                        {0.1, 10, 35},
                        {3.0, 10, 30},
                        {0.1, 10, 35}}

local P = -torch.Tensor{{.3689, .1170, .2673},
                        {.4699, .4387, .7470},
                        {.1091, .8732, .5547},
                        {.0381, .5743, .8828}} 

local a = -torch.Tensor{{1.0, 1.2, 3.0, 3.2}}
------------------------------------------------
--                                    hartmann3
------------------------------------------------
function hartmann3(X)
  if (X:dim() == 1 or X:size(1) == X:nElement()) then
    X = X:resize(1, X:nElement())
  end
  assert(X:size(2) == 3)

  -------- Compute Hartmann3 Function
  local N, xDim = X:size(1), X:size(2)
  local Y = torch.Tensor(N, 1)
  for n = 1,N do 
    Y[n] = torch.mm(a, torch.cmul(A, X[n]:repeatTensor(4, 1):add(P):pow(2)):sum(2):exp())
  end
  return Y
end

