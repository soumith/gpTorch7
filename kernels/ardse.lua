------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Automatic Relevance Determination Squared Exponential
(ARDSE) kernel class. 

Hyperparameters:
  1-by-(D+1) [amplitude, lengthscales]

To Do:
  - Debug hyperpriors
  - Run speedtests
  - Improve memory control

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-24
--]]

---------------- External Dependencies
local utils = require 'gp.utils'

------------------------------------------------
--                                         ARDSE
------------------------------------------------
local ARDSE, parent = torch.class('gp.kernels.ardse', 'gp.kernels.metakernel')

function ARDSE:__init()
  parent.__init(self)
end

function ARDSE.num_hypers(X, Y)
  return 1 + X:size(2)
end

function ARDSE.make_hypers(X, Y)
  local nHyp = 1 + X:size(2)
  local hyp  = torch.ones(nHyp, 1)
        hyp[1]:fill(Y:std())
  return hyp
end

function ARDSE.feasible_hypers(hyp)
  if not hyp then return true end

  -------- Amplitude
  local amp2     = utils.as_val(hyp[1])
  local amp2_min = 0
  local amp2_max = torch.exp(10) -- sufficiently large
  if amp2 <= amp2_min or amp2 >= amp2_max then
    return false
  end

  -------- Lengthscales
  local ls_min  = 0
  local ls_max  = 2 -- Top hat prior
  local ls      = hyp:sub(2, hyp:nElement())
  if ls:le(ls_min):any() or ls:ge(ls_max):any() then
    return false
  end

  return true
end


function ARDSE.cov(hyp, X, Z)
  local amp2 = utils.as_val(hyp[1])
  local ls   = hyp:sub(2, hyp:nElement())
  local cov  = utils.pdist(X, Z, 2, ls) -- squared distance

  if Z then -- amplify
    cov:mul(-0.5):exp():mul(amp2) 
  else -- stabilize & amplify 
    local eps = 1e-6
    cov:mul(-0.5):exp():add(torch.eye(cov:size(1)):clone():mul(eps)):mul(amp2)
  end

  collectgarbage()
  return cov
end

function ARDSE.derivatives(hyp, X, Z, cov)
  local N, xDim  = X:size(1), X:size(2)
  local nHyp     = hyp:nElement()
  local cov      = cov or ARDSE.cov(hyp, X, Z)
  local partials = torch.zeros(N, N, nHyp) -- fill via copy vs. add?

  -------- Log amplitude partial derivatives
  partials:select(3, 1):copy(cov)
  
  -------- Log lengthscale partial derivatives
  local dist2 = nil
  if Z then
    for idx = 1, xDim do
      dist2 = utils.pdist(X:select(2, idx):clone():resize(N,1), 
                          Z:select(2, idx):clone():resize(N,1), nil, 2, hyp[idx+1])
      partials:select(3, idx+1):copy(torch.cmul(cov, dist):mul(0.5))
    end
  else
    for idx = 1, xDim do
      dist2 = utils.pdist(X:select(2, idx):clone():resize(N,1), nil, 2, hyp[idx+1])
      partials:select(3, idx+1):copy(torch.cmul(cov, dist2):mul(0.5))
    end
  end

  collectgarbage()
  return partials
end

function ARDSE.gradients(hyp, X, resid, cache)
  -------- Compute intermediate terms
  local nHyp    = hyp:nElement()
  local cov     = cache.cov or ARDSE.cov(hyp, X, Z)
  local shared  = cache.shared
  if not shared then
    local N     = X:size(1)
    local invK  = cache.invK or torch.inverse(cov) -- could be wrong; lacks nzModel!
    local alpha = cache.alpha or torch.mm(invK, resid) -- alpha := K^(-1)(y - mu)
         shared = (torch.mm(alpha, alpha:clone():t()) - invK):t():resize(N, N, 1)-- better name?
  end

  -------- Compute gradients w.r.t. kernel parameters
  local grads = ARDSE.derivatives(hyp, X, nil, cov) -- partial derivatives
        grads = grads:cmul(shared:expandAs(grads)):sum(2):sum(1):resize(nHyp,1):mul(0.5)

  -------- Factor in hyperprior 
  grads = grads + ARDSE.grad_hyperprior(hyp)

  collectgarbage()
  return grads
end


function ARDSE.hyperprior(hyp)
   -------- Amplitude: Zero-mean log normal prior
   local hyperprior = -0.5*torch.log(utils.as_val(hyp[1]))^2

   -------- Lengthscales: Top-hat prior
   if torch.gt(hyp:sub(2,hyp:nElement()),2.0):any() then
     hyperprior = hyperprior - math.huge -- Note: math.huge*0.0 := nan
  end

  return hyperprior
end

------------------------------------------------
--                           Developer's Section
------------------------------------------------

-- function ARDSE.hyperprior(hyp)
--    -------- Amplitude: Zero-mean log normal prior
--    local hyperprior = -0.5*torch.log(utils.as_val(hyp[1]))^2

--    -------- Lengthscales: Top-hat prior
--    if torch.gt(hyp:sub(2,hyp:nElement()),2.0):any() then
--      hyperprior = hyperprior - math.huge -- Note: math.huge*0.0 := nan
--   end

--    return hyperprior
-- end

-- function ARDSE.grad_hyperprior(hyp)
--  local nHyp  = hyp:nElement()
--  local grads = torch.zeros(nHyp, 1)

--  -------- Amplitude gradient of zero-mean log normal prior
--  local amp2 = utils.as_val(hyp[1])
--  grads[1]:fill(-torch.log(amp2))

--  -------- Lengthscales: gradient of Top-hat prior
--  local ls   = hyp:sub(2, nHyp)
--  grads:sub(2, nHyp):copy(torch.eq(hyp:sub(2,nHyp), 2.0):mul(-math.huge))

--    return grads
-- end


return ARDSE
