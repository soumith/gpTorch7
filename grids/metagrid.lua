------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for gpTorch7 grids.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-18
--]]

---------------- External Dependencies
------------------------------------------------
--                                      metagrid
------------------------------------------------
local metagrid = torch.class('grids.metagrid')

function metagrid:__init()
end

function metagrid:__call__(config)
  local config = config or self.config
  return self.generate(config)
end

function metagrid.generate(config)
  print('Error: generate() method not implemented')
end



