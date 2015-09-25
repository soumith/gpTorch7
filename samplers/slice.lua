------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Slice sampler for gpTorch7.

Authored: 2015-09-16 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
local utils = gpTorch7.utils

------------------------------------------------
--                                         slice
------------------------------------------------
local sampler, parent = torch.class('samplers.slice', 'samplers.metasampler')

function sampler:__init()
  parent.__init(self)
end

function sampler:__call__(f, X0, opt, f_args)
  local opt     = sampler.configure(opt)
  local samples = sampler.sample(f, X0, opt, f_args)
  return samples
end

---------------- Default Settings
function sampler.configure(opt)
  local opt       = opt or {}
  opt['max_step'] = opt.max_step or 1e3
  opt['nSamples'] = opt.nSamples or 1

  -------- 'Step out' to generate envolope
  if opt.step_out ~= false then
    opt['step_out'] = true
  end

  -------- Implicit f := log(f(x))
  if opt.logspace ~= false then
    opt['logspace'] = true
  end

  -------- Exponentiate sampled values X
  if opt.expon ~= false and opt.logspace then
    opt['expon'] = true
  end
  return opt
end

---------------- Execute slice sampler
function sampler.sample(f, X0, opt, f_args)
  local X0 = X0:clone():repeatTensor(opt.nSamples, 1)
  local N, xDim = X0:size(1), X0:size(2)
  local samples = torch.Tensor(N, xDim)
  local x0      = torch.Tensor(1, xDim)
  
  -------- Gibbs slice sampler
  if opt.gibbs then
    local x1  = torch.zeros(1, xDim)
    local dir = torch.zeros(1, xDim)
    local order, d

    for n = 1,N do
      x0    = X0[{{n},{}}] -- maintains 1-by-xDim
      order = torch.randperm(x0:size(2)):long()
      for itr = 1,xDim do
        d = order[itr]
        dir[1][d] = 1.0
        x1[1][d]  = sampler.directed_slice(opt, f, f_args, dir, x0)[1][d]
        dir[1][d] = 0.0
        collectgarbage()
      end
      samples[n] = x1
      x1:fill(0.0)
    end

  -------- Vanilla slice sampling
  else
    for n = 1,N do
      x0  = X0[{{n},{}}] -- maintains 1-by-xDim
      dir = torch.randn(1, xDim)
      dir = dir:div(dir:norm())
      samples[n] = sampler.directed_slice(opt, f, f_args, dir, x0)
      collectgarbage()
    end
  end

  ---------------- Postprocessing
  if opt.expon then
    samples = samples:exp()
  end

  return samples
end

---------------- Slice sampler w/ specified direction
function sampler.directed_slice(opt, f, f_args, dir, x0)
  local N, xDim = x0:size(1), x0:size(2)

  local stepsize = opt.widths or torch.Tensor(1, xDim):fill(opt.width or 1.0)
  local itr, idx = 0, 0 -- counter and index variables


  -------- Perturbation function for f
  local function f_dx(dx)
    local dx = dx or torch.zeros(1, xDim)
    return f(x0 + dir:clone():cmul(dx), f_args)
  end

  -------- Draw sample Y ~ [0, f(x)]
  local Y = f_dx()
  if opt.logspace then
    Y = Y + torch.log(torch.rand(1))[1]  -- same as: log(f(x) * rand()) 
  else
    Y = Y * torch.rand(1)[1]
  end

  -------- Construct horizontal envelope 
  local right = torch.rand(1, xDim):cmul(stepsize)
  local left  = right - stepsize

  -------- Stepping out
  if opt.step_out then
    itr = 0 
    while f_dx(right) > Y and itr < opt.max_step do
      itr   = itr + 1
      right = right + stepsize
    end
    
    itr = 0
    while f_dx(left) > Y and itr < opt.max_step do
      itr  = itr + 1
      left = left - stepsize
    end
  end

  -------- Stepping in
  local dx, y, itr = 0, 0, 0
  while true do
    itr = itr + 1
    dx  = left + (right - left)*torch.rand(1)[1]
    y   = f_dx(dx)

    if y ~= y then -- NaN condition
      print('Error: samplers.slice encountered a NaN')
      break
    end

    if y > Y then -- Acceptance condition
      break
    end

    if dx:eq(0.0):any() then
      print('Error: samplers.slice shrank to zero')
      break
    end

    idx = dx:gt(0):nonzero() -- Update right
    if idx:dim() > 0 then
      utils.indexCopy(right, dx, idx:select(2,2))
    end

    idx = dx:lt(0):nonzero() -- Update left
    if idx:dim() > 0 then
      utils.indexCopy(left, dx, idx:select(2,2))
    end

    collectgarbage()
  end
  
  collectgarbage()
  return x0 + dir:clone():cmul(dx)
end