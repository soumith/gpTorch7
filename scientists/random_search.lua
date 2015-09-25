------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Random search scientist class for gpTorch7.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-25
--]]

---------------- External Dependencies
paths = require('paths')
math  = require('math')
local utils = gpTorch7.utils
------------------------------------------------
--                                 random_search
------------------------------------------------
local scientist, parent = torch.class('scientists.random_search', 
                                      'scientists.metascientist')

function scientist:__init(config, objective)
  parent.__init(self)

  -------- Establish settings
  local config   = scientist:configure(config)
  self.config    = config
  self.objective = objective

  -------- Generate Candidates
  local grid      = gpTorch7.grids[config.grid.type]()
  self.candidates = grid(config.grid)

  -------- Preestablish random ordering
  self.order      = torch.randperm(config.grid.size)
  self.nTrials     = 0

  -------- Allocate result tensors
  self.best       = {}
  self.best['x']  = torch.Tensor(1, config.xDim)
  self.best['y']  = torch.Tensor(config.yDim):fill(math.huge)
end

function scientist:configure(config)
  local config = config
  if config then
    config = utils.deepcopy(config)
  else
    config = {}
  end

  -------- Scientist
  config['verbose']  = config.verbose or 0
  config['budget']   = config.budget or 500
  config['msg_freq'] = config.msg_freq or 1
  config['yDim']     = config.yDim or 1

  -------- Grid
  local grid     = config.grid or {}
  grid['type']   = grid.type or 'random'
  grid['size']   = grid.size or 2e3
  grid['dims']   = grid.dims or config.xDim
  grid['mins']   = grid.mins or torch.zeros(1, grid.dims)
  grid['maxes']  = grid.maxes or torch.ones(1, grid.dims)
  config['grid'] = grid
    
  -------- Model
  config['model'] = {}

  return config
end

function scientist:eval()
  -- random search; so, do nothing
end

---------------- Nominate a candidate 
function scientist:nominate()
  return self.candidates:select(1, self.order[self.nTrials]):resize(1, self.config.xDim)
end

function scientist:run_trial()
  -------- Increment trial counter
  self.nTrials = self.nTrials + 1
  local x = self:nominate()
  local y = self.objective(x)
  return x, y
end

function scientist:run_experiment()
  local x, y
  for t = 1, self.config.budget do
    -------- Perform a single trial 
    x,y = self:run_trial()

    -------- Store results
    self:update_best(x,y)

    -------- Display results
    self:progress_report(t, x, y)
  end
end

function scientist:update_best(x, y)
  if self.best.y:gt(y):all() then
    self.best.t = self.nTrials
    self.best.x = x
    self.best.y = y
  end
end

function scientist:progress_report(t, x, y)
  if t % self.config.msg_freq == 0 then
    local config = self.config
    local msg = string.format('Trial: %d of %d', t, config.budget)
    print('================================================')
    print(string.rep(' ', 48-msg:len()).. msg)
    print('================================================')
    print('> Most recent:')
    print('Hypers: ' .. utils.tnsr2str(x,','))
    print('Response: ' .. utils.tnsr2str(y))
    print(string.format('> Best seen (#%d):', self.best.t))
    print('Hypers: ' .. utils.tnsr2str(self.best.x, ','))
    print('Response: ' .. utils.tnsr2str(self.best.y) .. '\n')
  end
end

