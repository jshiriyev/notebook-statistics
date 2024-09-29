from matplotlib import pyplot as plt

import numpy as np

# from borepy.scomp.regression import LinearSimple

from scipy.stats import linregress

from scipy.stats import t as tstat

data = np.loadtxt("table_11_1.txt",skiprows=3)

x,y = data.T

def regress(x,y):

	class result:
		pass

	n = x.size

	df = n-2

	linear = linregress(x,y)

	Sxx = np.nansum((x-np.nanmean(x))**2)

	ycap = linear.slope*x+linear.intercept

	s2 = np.sum((ycap-y)**2)/df

	merr = np.sqrt(s2/Sxx)
	berr = np.sqrt(s2/n/Sxx*np.nansum(x**2))

	result.df = df
	result.linear = linear
	result.merr = merr
	result.berr = berr

	return result

def percentile(result,prc=0.5):

	class model:
		pass

	model.Di = result.linear.slope+tstat.ppf(prc,result.df)*result.merr
	model.yi = result.linear.intercept+tstat.ppf(prc,result.df)*result.berr

	return model

result = regress(x,y)

model = percentile(result,1-0.025)

print(model.Di,model.yi)

print(result.linear.intercept)
print(3.2295*np.sqrt(41086/33/4152.18))

# x = data[:,0]
# y = data[:,1]

# slr = LinearSimple(y,x)

# slr.train()

# print(f"{slr.b0} intercept")
# print(f"{slr.b1} slope")
# print(f"{slr.R2} R2")

# print(f"{slr.beta0confint(0.05)} beta0 confidence interval")
# print(f"{slr.beta1confint(0.05)} beta1 confidence interval")

# beta00 = 0
# beta10 = 0.9

# print(f"{slr.beta0test(beta00)} test {beta00 = }")
# print(f"{slr.beta1test(beta10)} test {beta10 = }")

# xest = np.linspace(0,60,100)
# yest = slr.estimate(xest)

# ymeanl,ymeanu = slr.meanconfint(xest,0.05)

# ypl,ypu = slr.predict(xest,0.05)

# plt.scatter(x,y)

# plt.plot(xest,yest,color="red")
# plt.plot(xest,ymeanl,color='red',linestyle='--',linewidth=0.5)
# plt.plot(xest,ymeanu,color='red',linestyle='--',linewidth=0.5)
# plt.plot(xest,ypl,color='blue',linestyle='--',linewidth=0.5)
# plt.plot(xest,ypu,color='blue',linestyle='--',linewidth=0.5)

# plt.xlim((0,54))
# plt.ylim((0,50))

# plt.xlabel("x-values")
# plt.ylabel("y-values")

# plt.show()