------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Gaussian processes regressor demo running on
synthetic data.

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
--         Gaussian Process Regressor (GPR) Demo
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

-------- Marginalize over hyperparameters via Monte Carlo integration
function run_demo()
	------ Sample hyperparameters (slice sampling)
	print('Sampling hyperparameters...')
	samples = gpr:sample_hypers(X, Y, nil, config_slice)
  print('> Sample Distribution: ')
  local mu = samples:mean(1):storage()
  local sd = samples:std(1):storage()
  for k = 1, samples:size(2) do
    print(string.format('hyp[%d] ~ N(%.2e, %.2e)', k, mu[k], sd[k]))
  end
  print(' ')

  print('Computing negative log-likelihoods...')
	-------- Compute initial NLL
	nll0 = -gpr:loglik(X, Y)
	print(string.format('> Initial  NLL: N(%.2e, %.2e) ', nll0, 0))

	-------- Compute posterior NLL
	nll1 = torch.Tensor(samples:size(1), 1)
	for s = 1, samples:size(1) do
		nll1[s] = -gpr:loglik(X, Y, samples[s])
	end
	print(string.format('> Marginal NLL: N(%.2e, %.2e)', nll1:mean(), nll1:var()))
end

-------- Options / Settings
opt = 
{
	N      = 100,
	xDim   = 3,
	yDim   = 1,
	nz 	   = math.sqrt(1e-6),	
}

config_gpr =
{
  kernel  = 'ardse',
  nzModel = 'GaussianNoise_iso',
  mean    = 'constant',
}

config_slice = {nSamples = 100}

X = nil
Y = nil

synthesize()
gpr = gpr_init(config_gpr, X, Y)


samples = nil
nll0 = nil
nll1 = nil

msg  = '================================================\n'..
       '                 Gaussian Process Regressor Demo\n'..
       '================================================\n'..
       'The present script compares the negative log-\n'..
       'likelihood of a synthetic dataset given a\n'..
       'generic initialization of the GP hyperparameters\n'..
       'versus that obtained when marginalizing over \n'..
       'said hyperparameters via Monte Carlo integration\n'..
       '(slice sampling).\n'
print(msg)
run_demo()


