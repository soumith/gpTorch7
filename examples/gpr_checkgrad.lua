------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Gaussian processes regressor gradient check.


Authored: 2015-09-12 (jwilson)
Modified: 2015-09-25
--]]

---------------- External Dependencies
torch = require('torch')
optim = require('optim')
math  = require('math')
paths = require('paths')
gpTorch7 = require('..')

------------------------------------------------
--    Gaussian Process Regressor (GPR) Checkgrad
------------------------------------------------

-------- Generate synthetic data
function synthesize()
  A = torch.rand(1, opt.xDim)
  X = torch.mm(torch.range(0, opt.N):div(opt.N):resize(opt.N, 1), A)
  H = torch.randn(opt.xDim, opt.yDim)
  Y = torch.mm(X, H) + torch.randn(opt.N,1):mul(opt.nz)
end

-------- Initialize GPR
function gpr_init(config, X, Y)
  gpr = gpTorch7.models.gp_regressor(config)
  gpr:init(X, Y)
  return gpr
end

-------- Perform gradient check
function run_checkgrad()
  print('Performing gradient check...')
  diff, exact, approx = gpr:checkgrad(X, Y)
  print('> Net deviation: ' .. diff)

  print('> Hyperparameter Specific Results:')
  print(' [Extact] [Finite]  [Ratio]')
  print(torch.cat(torch.cat(exact, approx), exact:clone():cdiv(approx)))
  return {diff=diff, exact=exact, approx=approx}
end

-------- Options / Settings
opt = 
{
  N      = 200,
  xDim   = 3,
  yDim   = 1,
  nz     = math.sqrt(1e-6), 
}

config_gpr =
{
  kernel  = 'ardse',
  nzModel = 'GaussianNoise_iso',
  mean    = 'constant',
}

X = nil
Y = nil

synthesize()
gpr = gpr_init(config_gpr, X, Y)

msg  = '================================================\n'..
       '            Gaussian Process Regressor Checkgrad\n'..
       '================================================\n'..
       'The present script performs a gradient check\n'..
       'w.r.t. the GP covariance function and noise\n'..
       'model hyperparameters.\n'
print(msg)

res = run_checkgrad()