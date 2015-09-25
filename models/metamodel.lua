------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for gpTorch7 models.

Authored: 2015-09-15 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
local utils = require 'gp.utils'
------------------------------------------------
--                                     metamodel
------------------------------------------------
local metamodel = torch.class('gp.models.metamodel')

function metamodel:__init()
end

function metamodel:save()
end

function metamodel:load()
end

function metamodel:update()
end

function metamodel:cache()
  local cache = 
  {
    config  = self.config,
    kernel  = self.kernel,
    nzModel = self.nzModel,
    mean    = self.mean,
    hyp     = self.hyp
  }
  return cache
end

function metamodel:__tostring__()
  return torch.type(self)
end

return metamodel
