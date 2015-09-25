------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Global optimization on benchmark functions.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-25
--]]

---------------- External Dependencies
paths     = require('paths')
gpTorch7  = require('..')

------------------------------------------------
--                                Initialization
------------------------------------------------

---------------- Argument Parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-verbose',   1, 'specify which print level')
cmd:option('-scientist', 'bo', 'specify which scientist to use: {bo, rs}')
cmd:option('-benchmark', 'braninhoo', 'specify which function to optimize: '..
                                      '{braninhoo, hartmann3, hartmann6}')

cmd:option('-xDim',  -1,  'specify input dimensionality for experiment')
cmd:option('-yDim',  1,   'specify output dimensionality for experiment')
cmd:option('-noisy',      false, 'specify observations as noisy')
cmd:option('-grid_size',  20000, 'specify size of candidate grid')
cmd:option('-mins', '',   'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',      '', 'specify maxima for inputs (defaults to 1.0)')
cmd:option('-score',      'ei', 'specify acquisition function to be used by scientist; {ei, ucb}')

cmd:text()
opt = cmd:parse(arg or {})

---------------- Experiment Configuration
local expt = 
{  
  sci   = opt.scientist,
  func  = opt.benchmark,
  xDim  = opt.xDim,
  yDim  = opt.yDim,
  model = {noiseless = not opt.noisy},
  grid  = {size = opt.grid_size}
}

---- Establish xDim for experiment
if     (opt.benchmark == 'braninhoo') then
  opt.xDim = 2
elseif (opt.benchmark == 'hartmann3') then
  opt.xDim = 3
elseif (opt.benchmark == 'hartmann6') then
  opt.xDim = 6
else
  print('Error: Unrecognized benchmark function specified; aborting...')
  return
end
expt.xDim, expt.grid['xDim'] = opt.xDim, opt.xDim

---- Establish minimia/maxima
if (opt.mins ~= '') then
  loadstring('opt.mins='..opt.mins)()
else
  opt.mins = torch.zeros(1, opt.xDim)
  expt.grid.mins = opt.mins
end

if (opt.maxes ~= '') then
  loadstring('opt.maxes='..opt.maxes)()
else
  opt.maxes = torch.ones(1, opt.xDim)
  expt.grid.maxes = opt.maxes
end

---- Choose acquistion function
if     (opt.score == 'ei') then
  expt.model['score'] = 'expected_improvement'
elseif (opt.score == 'ucb') then
  expt.model['score'] = 'confidence_bound'
end


------------------------------------------------
--                                 run_benchmark
------------------------------------------------
function run_benchmark(expt)

  -------- Load benchmark function
  paths.dofile('benchmarks/' .. expt.func .. '.lua')

  -------- Initialize scientist
  local scientist
  if expt.sci == 'bo' then
    scientist = gpTorch7.scientists.bayesopt(expt, _G[expt.func])
  elseif expt.sci == 'rs' then
    scientist = gpTorch7.scientists.random_search(expt, _G[expt.func])
  else
    print('Error: Unrecognized scientist specified; aborting...')
    return
  end

  -------- Perform experiment
  scientist:run_experiment()
end

run_benchmark(expt)