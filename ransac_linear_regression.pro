;+
; :Description:
;    Wrapper to apply the RANSAC implementation by Carl Salvaggio to the problem of univariate linear regression.
;
; :Params:
;    x: vector of the independent variable
;    y: vector of the dependent variable
;
; :Keywords:
;    minimumSamples: the minimum number of points required to fit the model (default 80% of the total).
;    distanceThreshold: the maximum residual from the determined model to be considered an inlier (default MAD).
;    probability_error_free_selection: the desired probability of choosing at least one
;       sample free from outliers (default=0.99)
;    inliers: a returned vector containing the indices corresponding to the inliers.
;    outliers: a returned vector containing the indices corresponding to the outliers.
;    
; :Requires:
;    RANDOM_SAMPLE
;    RANSAC by Carl Salvaggio
;    To install them, prompt:
;    IDL> ipm, /install, 'https://github.com/csalvaggio/IDL_RIT_Salvaggio'
;    
; :Author: Emanuele Mandanici, DICAM
; 
; :Date: 12 November 2022
;-
function ransac_linear_regression, x, y, minimumSamples=mins, distanceThreshold=dthr, $
  probability_error_free_selection=prob, inliers=inl, outliers=outl

  ; check input
  nx = n_elements(x)
  if nx ne n_elements(y) then return, !null
  ; setting default values for the mandatory arguments of Salvaggio's routine
  if ~keyword_set(mins) then mins = fix(nx*0.8)
  if ~keyword_set(dthr) then dthr = MAdeviation(y)

  ; input preparation
  data = fltarr(2,nx)
  data[0,*] = x
  data[1,*] = y

  ; Salvaggio's function call
  best = ransac(data,'linearfit','fitdistance',mins,dthr,probability=prob,inliers=inl)
  ; identification of outliers
  outl = indexcomplement(inl,nx)

  ; final calculation with inliers
  param = call_function('linearfit', data[*,inl], status)

  ; return
  if status then return, param else return, best
end

function linearfit, data, status
; wrapper for the linfit IDL function
  status = 0
  coeff = linfit(data[0,*],data[1,*]) ; data[2,samples]
  if total(finite(coeff)) eq 2 then status = 1
  return, coeff
end

function fitdistance, model, data, threshold
; measure error after the linear fit
  ym = model[0] + model[1] * data[0,*]
  diff = abs(ym - data[1,*])
  return, where(diff le threshold)
end

function MAdeviation, y
; median absolte deviation
  m = median(y)
  ad = abs(y-m)
  return, median(ad)
end

function indexcomplement, indices, n
; outliers identification
  if n_elements(indices) ge n then return,!null
  compl = -1
  for i = 0, n-1 do begin
    if where(indices eq i) eq -1 then compl = [compl, i]
  endfor
  return, compl[1:-1]
end

pro test_ransac
; silly test of the RANSAC wrapper
  ; sample generation
  x = findgen(20)
  y = 2 * x + 1
  ; outliers
  y[1] = 0.0
  y[7] = 50.0
  y[12] = 1.0
  y[18] = 150.0
  
  best = ransac_linear_regression(x,y,inliers=inl,outliers=outl,distance=2)

  print, 'normal fit = ', linfit(x,y)
  print, 'ransac fit = ', best
  print, 'inliers: ', inl
  print, 'outliers: ', outl
end