------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Utility methods for gpTorch7.

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-25

--]]

---------------- External Dependencies
require('torch')
require('math')

---------------- Constants
local sqrt2_inv   = 1/math.sqrt(2)
local sqrt2pi_inv = 1/math.sqrt(2*math.pi)

------------------------------------------------
--                                gpTorch7_utils
------------------------------------------------
local utils = {}
--------------------------------
--         Imports from penlight
--------------------------------
utils.find     = require('pl.tablex').find
utils.tbl_size = require('pl.tablex').size
utils.deepcopy = require('pl.tablex').deepcopy



--------------------------------
--              Tensor to String
--------------------------------
function utils.tnsr2str(tnsr, delim, format, align)
  local delim  = delim  or ''
  local format = format or '%.2e'
  local align  = align  or 'horiz'
  local tnsr   = tnsr
  local nDims, nRow, nCol = tnsr:dim(), tnsr:size(1), tnsr:size(2)

  if (nDims == 0) then
    print('0-dimensional tensors cannot be converted to string.')
    return
  end
  if (nDim == 1) then
    if align == 'vert' then
      tnsr = tnsr:clone():resize(tnsr:nElement(), 1)
    else
      tnsr = tnsr:clone():resize(1, tnsr:nElement())
    end
  end
  if (nDims > 2)  then
    if nRow*nCol == tnsr:nElement() then
      tnsr = tnsr:clone():resize(nRow, nCol)
    else
      print('Support for >2 tensors not currently available for tnsr2str()')
      return
    end
  end
  
  local str  = ''
  for row = 1, nRow do
    for col = 1, nCol do  
     str = str .. string.format(format, tnsr[row][col])
     if col < nCol then str = str .. delim .. ' ' end
    end
    if row < nRow then str = str .. '\n' end
  end
  -- for k = 1,N do
  --   str = str .. string.format(format, store[k])
  --   if k < N then str = str .. delim .. ' ' end
  -- end
  return str
end
--------------------------------
--        Return target as value
--------------------------------
-- NEEDS WORK
function utils.as_val(x, idx)
  local idx = idx or 1
  if torch.isTensor(x) then
    return x:clone():storage()[idx] -- sloppy
  else
    return x
  end
end

--------------------------------
--           Return shape tensor
--------------------------------
function utils.shape(tnsr)
   local shape = torch.LongTensor(tnsr:dim())
         shape:storage():copy(tnsr:size())
  return shape
end

--------------------------------
--    Extract slices from tensor
--------------------------------
function utils.slice(tnsr, idx, dim)
  -------- Short circuit
  if idx:nElement() == 0 then
    return torch.Tensor()
  end

  -------- Format index tensor 
  local idx = idx
  local dim = dim or 1
  if idx:dim() == 1 or idx:size(2) > 1 then
    idx = idx:clone():resize(idx:nElement(), 1)
  end
  -------- Create coordinates for gather
  idx = idx:repeatTensor(1, tnsr:nElement()/tnsr:size(dim))
  return tnsr:gather(dim, idx)
end


--------------------------------
--        Safer append to tensor
--------------------------------
function utils.append(tnsr, subtnsr, axis)
  if torch.isTensor(tnsr) and tnsr:dim() > 0 then
    tnsr = torch.cat(tnsr, subtnsr, axis)
  else
    tnsr = subtnsr:clone()
    if axis == 1 and (tnsr:dim() == 1 or tnsr:size(1) == tnsr:nElement()) then
      tsnr:resize(1, tnsr:nElement())  -- hack to get desired shape
    end
  end
  return tnsr
end

--------------------------------
--     Remove slices from tensor
--------------------------------
function utils.remove(tnsr, idx, axis)
  -------- Determine elements maintained by tnsr
  local axis  = axis or 1
  local Ns    = tnsr:size(axis)
  local keep  = torch.range(1, Ns, 'torch.LongTensor')
        keep:indexFill(1, idx:select(2,1), 0)
  return utils.slice(tnsr, keep:nonzero())
end

--------------------------------
-- Move slice(s) from src to res
--------------------------------
function utils.steal(res, src, idx, axis_r, axis_s)
  local res = res
  local src = src
  local idx = idx

  local axis_r = axis_r or 1
  local axis_s = axis_s or 1

  if idx:dim() == 1 then
    idx = idx:clone():resize(1, idx:nElement())
  end

  -------- Append slices to res
  res = utils.append(res, utils.slice(src, idx, axis_s), axis_r)

  -------- Remove slices from src
  src = utils.remove(src, idx, axis_s)

  collectgarbage()
  return res, src
end

--------------------------------
--  Wrapper for tensor.indexCopy
--------------------------------
function utils.indexCopy(res, src, idx, axis_r, axis_s)
  local axis_r = axis_r or res:dim()
  local axis_s = axis_s or src:dim()
  return res:indexCopy(axis_r, idx, src:index(axis_s, idx))
end

--------------------------------
--             Kronecker Product
--------------------------------
function utils.kron(X, Z, buffer)
  assert(X:dim() == 2 and Z:dim() == 2) -- temp hack, should generalize this
  local N, M = X:size(1), X:size(2)
  local P, Q = Z:size(1), Z:size(2)
  local K    = buffer or torch.Tensor(N*P, M*Q)
  for row = 1,N do
    for col = 1,M do
      K[{{(row - 1)*P + 1, row*P},{(col - 1)*Q + 1, col*Q}}]
          = torch.mul(Z, X[row][col])
    end
  end
  return K
end

--------------------------------
--              Tensor Transpose
--------------------------------
function  utils.transpose(tnsr, inplace)
  local nDim = tnsr:dim()

  if inplace then res = tnsr
  else res = tnsr:clone() end

  for d = 1, math.floor(nDim/2) do
    res = res:transpose(d, nDim-d+1):clone()
  end
  collectgarbage()
  return res
end

--------------------------------
--               Tensor Reversal
--------------------------------
function utils.flip(tnsr, axis, idx0, idx1, inplace)
  local axis = axis or 1
  local idx0 = idx0 or 1
  local idx1 = idx1 or tnsr:size(axis)
  if inplace then
    tnsr = tnsr:index(axis, torch.range(-idx1, -idx0, 'torch.LongTensor'):mul(-1))
    return tnsr
  else
    return tnsr:index(axis, torch.range(-idx1, -idx0, 'torch.LongTensor'):mul(-1))
  end
end

--------------------------------
--               Linear Indexing
--------------------------------
function utils.linear_index(shape, coords, order)
  local nDim   = shape:nElement()
  local order  = order or 'C'

  local coords = coords
  if (coords:dim() == 1) then
    coords = coords:clone():resize(coords:nElement(), 1)
  end

  local offset
  if order == 'C' then
    offset = utils.flip(torch.cat(torch.ones(1, 'torch.LongTensor'),
                        utils.flip(shape, 1, 2):cumprod(), 1))
  elseif order == 'F' then
    offset = torch.cat(torch.ones(1, 'torch.LongTensor'), shape:sub(1, nDim-1), 1):cumprod()
  end
  return torch.mv(coords - 1, offset):add(1)
end

--------------------------------
--   Linear Indices of lower tri
--------------------------------
function utils.tril_indices(N)
  local idx   = torch.LongTensor((N^2+N)/2)
  local count = 1
  for row = 1,N do
    idx[{{count, count+row-1}}] = torch.range(1, row):add(N*row-N)
    count = count + row
  end
  return idx
end

--------------------------------
--   Linear Indices of upper tri
--------------------------------
function utils.triu_indices(N)
  local idx   = torch.LongTensor((N^2+N)/2)
  local count = 1
  for row = 0,N-1 do
    idx[{{count, count+N-row-1}}] = torch.range(1+row, N):add(N*row)
    count = count + N - row
  end
  return idx
end

--------------------------------
--          Tensor Vectorization
--------------------------------
function utils.vect(tnsr, order, inplace)
  local N, nDim = tnsr:nElement(), tnsr:dim()
  local order   = order or 'F'
  local shape   = utils.shape(tnsr)

  -------- noop, already vectorized
  if (shape:max() == N) then return tnsr end

  -------- Vectorize in-place
  if inplace then res = tnsr
  else res = tnsr:clone() end

  -------- Vectorize using C indexing convention 
  -- Row-major order: Indices of last dimension change first.
  -- Tensors are natively row-major; so, just call resize().
  if order == 'C' then
    res:resize(N,1)

  -------- Vectorize using Fortran indexing convention
  -- Column-major order: Indices of leading dimension change first
  elseif order == 'F' then
    utils.transpose(res, true)
    res:resize(N,1)
  end

  collectgarbage()
  return res
end

--------------------------------
--             Pairwise Distance
--------------------------------
---- Can this be sped up / improved upon?
function utils.pdist(X, Z, p, lenscale, w_root)
  local p    = p or 2
  local dist = nil

  -------- Compute pairwise lp distance (w/o root)
  if lenscale then
    ---- Inverse Lengthscales
    local inv_ls = torch.ones(lenscale:size()):cdiv(lenscale)
    if (inv_ls:dim() == 1) then
      inv_ls:resize(inv_ls:size(1), 1)
    end
    if Z then
      local M, N = X:size(1), Z:size(1)
      local X_ss = torch.mm(X:clone():pow(p), inv_ls):repeatTensor(1, N)
      local Z_ss = torch.mm(Z:clone():pow(p), inv_ls):repeatTensor(1, M)
      dist = Z:clone():t()
      dist:cmul(inv_ls:expandAs(dist))
      dist = torch.zeros(M, N):mm(X, dist):mul(-2.0):add(X_ss):add(Z_ss:t())
    else
      local N    = X:size(1)
      local X_ss = torch.mm(X:clone():pow(p), inv_ls):repeatTensor(1, N)
      dist = X:clone():t()
      dist:cmul(inv_ls:expandAs(dist))
      dist = torch.zeros(N, N):mm(X, dist):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  else
    if Z then
      local M, N = X:size(1), Z:size(1)
      local X_ss = torch.sum(X:clone():pow(p), 2):repeatTensor(1, N)
      local Z_ss = torch.sum(Z:clone():pow(p), 2):repeatTensor(1, M)
      dist = torch.zeros(M, N):mm(X, Z:t()):mul(-2.0):add(X_ss):add(Z_ss:t())
    else
      local N    = X:size(1)
      local X_ss = torch.sum(X:clone():pow(p), 2):repeatTensor(1, N)
      dist = torch.zeros(N, N):mm(X, X:t()):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  end

  -------- Restrict to be non-negative (numerical stability hack)
  dist:clamp(0.0, math.huge)
  collectgarbage()

  if w_root then return dist:pow(1.0/p)
  else           return dist end
end

--------------------------------
--      Error Function (Approx.)
--------------------------------
function utils.erf(x)
   -------- Constants
  local c1, c2 = 0.254829592, -0.284496736
  local c3, c4 = 1.421413741, -1.453152027
  local c5, p  = 1.061405429,  0.3275911

  -------- Argument Parsing
  if not torch.isTensor(x) then
    local t = type(x)
    if t == 'table' then
      x = torch.Tensor(x)
    elseif t == 'number' then
      x = torch.Tensor{x}
    end
  end

  -------- Error Function
  local sign = x:ge(0.0):type(x:type()):mul(2):add(-1)
  local x    = torch.abs(x)
  local t    = x:clone():mul(p):add(1):pow(-1)
  local erf  = t:clone():mul(c5):add(c4):cmul(t):add(c3)
                :cmul(t):add(c2):cmul(t):add(c1):cmul(t)
                :cmul(x:pow(2):mul(-1.0):exp())
                :mul(-1):add(1):cmul(sign)
  return erf
end

--------------------------------
--           Standard Normal PDF
--------------------------------
function utils.standard_pdf(x)
  return torch.exp(x:pow(2):mul(-0.5)):mul(sqrt2pi_inv)
end

--------------------------------
--           Standard Normal CDF
--------------------------------
function utils.standard_cdf(x)
  return utils.erf(x:mul(sqrt2_inv)):add(1):mul(0.5)
end

return utils

