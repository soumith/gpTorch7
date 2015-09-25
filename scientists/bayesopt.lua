------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Bayesian Optimization scientist class for gpTorch7.

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
local scientist, parent = torch.class('scientists.bayesopt',
                                      'scientists.metascientist')

function scientist:__init(config, objective)
  parent.__init(self)

  -------- Establish settings
  local config   = scientist:configure(config)
  self.config    = config
  self.objective = objective
  self.nTrials   = 0

  -------- Generate Candidate Grid
  local grid      = gpTorch7.grids[config.grid.type]()
  self.candidates = grid(config.grid)

  self.responses  = nil
  self.observed   = nil

  -------- Initialize model / acquisition function
  self.model = gpTorch7.models[config.model.type](config.model)
  self.score = gpTorch7.scores[config.score]()

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

  ---------------- Default Settings (patch to use pl.tablex.update)
  -------- Scientist
  config['verbose']  = config.verbose or 0
  config['budget']   = config.budget or 500
  config['msg_freq'] = config.msg_freq or 1
  config['score']    = config.score or 'expected_improvement'
  config['nInitial'] = config.nInitial or 2
  config['nSamples'] = config.nSamples or 10
  

  -------- Grid
  local grid     = config.grid or {}
  grid['type']   = grid.type or 'random'
  grid['size']   = grid.size or 2e3
  grid['dims']   = grid.dims or config.xDim
  grid['mins']   = grid.mins or torch.zeros(1, grid.dims)
  grid['maxes']  = grid.maxes or torch.ones(1, grid.dims)
  config['grid'] = grid
    
  -------- Model
  local model      = config.model or {}
  model['type']    = model.type or 'gp_regressor'
  model['kernel']  = model.kernel or 'ardse'
  model['nzModel'] = model.nzModel or 'GaussianNoise_iso'
  model['mean']    = model.mean or 'constant'
  model['sampler'] = model.sampler or 'slice'
  config['model']  = model

  return config
end

function scientist:eval(candidates)
  -------- Local aliases
  local X_obs   = self.observed
  local Y_obs   = self.responses
  local X_hid   = self.candidates
  local X_pend  = self.pending

  -------- Assess acquisition function while marginalizing
  -- over hyperparameters via MC integration
  local samples  = self.model:sample_hypers(X_obs, Y_obs)
  local score    = torch.zeros(X_hid:size(1))
  local nSamples = self.config.nSamples
  local hyp      = torch.Tensor()

  for s = 1,nSamples do
    hyp = self.model:sample_hypers(X_obs, Y_obs, nil, nil, true)
    hyp = self.model:parse_hypers(hyp)
    score:add(self.score(self.model, hyp, X_obs, Y_obs, X_hid))
    collectgarbage()
  end
  score:div(nSamples)
  return score
end

---------------- Nominate a candidate
function scientist:nominate(candidates)
  local candidates = candidates or self.candidates
  local idx

  -------- Select initial points randomly
  if self.nTrials <= self.config.nInitial then
    idx = torch.rand(1):mul(candidates:size(1)):long()

  -------- Nominate according to acquistion values
  else
    local score = self:eval(candidates)
    min, idx  = score:max(1)
  end

  return idx
end

function scientist:run_trial()
  -------- Increment trial counter
  self.nTrials = self.nTrials + 1

  -------- Nominate candidate and mark as pending
  local idx    = self:nominate()
  self.pending, self.candidates =  utils.steal(self.pending, self.candidates, idx)

  idx      = self.pending:size(1) -- update idx
  nominee  = self.pending:select(1, idx)

  -------- Pass nominee to blackbox
  local y  = self.objective(nominee)

  -------- Format response value (y)
  if not torch.isTensor(y) then
    if type(y) == 'number' then
      y = torch.Tensor{{y}}
    elseif type(y) == 'table' then
      y = torch.Tensor(y)
    end
  end
  if y:dim() == 1 then y:resize(y:nElement(), 1) end

  -------- Store result and mark nominee as observed
  if self.nTrials == 1 then
    self.responses = y
  else
    self.responses = self.responses:cat(y, 1)
  end
  self.observed, self.pending = 
    utils.steal(self.observed, self.pending, torch.LongTensor{idx})

  -------- Initialize model w/ values
  if self.nTrials == self.config.nInitial then
    self.model:init(self.observed, self.responses)
  end

  return nominee, y
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