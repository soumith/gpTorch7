------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for gpTorch7 samplers.

Authored: 2015-09-16 (jwilson)
Modified: 2015-09-21
--]]

---------------- External Dependencies
------------------------------------------------
--                                   metasampler
------------------------------------------------
local sampler = torch.class('samplers.metasampler')

function sampler:__init()
end

function sampler:__call__(f, X0, opt, f_args)
  local opt     = sampler.configure(opt)
  local samples = sampler.sample(f, X0, opt, f_args)
  return samples
end

---------------- Default Settings
function sampler.configure(opt)
  local opt = opt or {}
  return opt
end

---------------- Sampling method
function sampler.sample()
  print('Error: sample() method not implemented')
end






