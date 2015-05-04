from StringIO import StringIO
import sys, os
import numpy as np

n_max = None
if len(sys.argv) > 1: n_max = int(sys.argv[1])

input = sys.stdin
pmf_filename = input.readline().strip() # stores pmf
rho_filename = input.readline().strip() # stores average density
bia_filename = input.readline().strip() # stores biased distribution
fff_filename = input.readline().strip() # stores F(i)
temperature = float(input.readline().strip())

xmin, xmax, delta, is_x_periodic = map(float, input.readline().strip().split())
nwin, niter, fifreq  = map(int, input.readline().strip().split())
tol = map(float, input.readline().strip().split())
is_x_periodic = bool(is_x_periodic)
nbin = int((xmax-xmin+0.5*delta)/delta)
kb = 0.0019872
kbt = kb * temperature
beta = 1.0/kbt

k1 = np.zeros(nwin)
cx1 = np.zeros(nwin)
var1 = np.zeros(nwin)
tseries = np.empty(nwin, dtype='S')
hist = np.zeros((nwin, nbin), dtype=np.int)
nb_data = np.zeros(nwin, dtype=np.int)
x1 = lambda j: xmin + (j+1)*delta - 0.5*delta

for i in range(nwin):
    fname = input.readline().strip()
    tseries[i] = fname
    cx1[i], k1[i] = map(float, input.readline().strip().split())

    def mkhist(fname, xmin, xmax, delta, ihist):
        xdata = []
        if os.path.exists(fname+'.gz'):
            import gzip
            fp = gzip.open(fname+'.gz')
        else:
            fp = open(fname)
        for line in fp:
            time, x = map(float, line.strip().split()[:2])
            xdata.append(x)
        x = np.array(xdata)
        xbins = [xmin+i*delta for i in range(nbin+1)]
        hist[ihist], edges = np.histogram(x, bins=xbins, range=(xmin, xmax))
        nb_data[ihist] = int(np.sum(hist[ihist,:]))

        print 'statistics for timeseries # ', ihist
        print 'minx:', '%8.3f' % np.min(x), 'maxx:', '%8.3f' % np.max(x)
        print 'average x', '%8.3f' % np.average(x), 'rms x', '%8.3f' % np.std(x)
        print 'statistics for histogram # ', ihist
        print int(np.sum(hist[ihist,:])), 'points in the histogram'
        print 'average x', '%8.3f' % (np.sum([hist[ihist,i]*(edges[i]+edges[i+1])/2 for i in range(nbin)])/np.sum(hist[ihist]))
        print

        var = 1.0/(nblock*(nblock-1))*np.sum([np.average((x[k:(k+1)*(len(x)/nblock)]-np.average(x))**2) for k in range(nblock)])
        return var

    var1[i] = mkhist(fname, xmin, xmax, delta, i)
    if debug: break


# write biased distribution
f = open(bia_filename, 'w')
for j in range(nbin):
    f.write("%8d\n" % np.sum(hist[:,j]))

# iterate wham equation to unbias and recombine the histogram
TOP = np.zeros(nbin, dtype=np.int32)
BOT = np.zeros(nbin)
rho = np.zeros(nbin)
V1 = np.zeros((nwin, nbin))
F = np.zeros(nwin)
F2 = np.zeros(nwin)
for i in range(nwin):
    for j in range(nbin):
        V1[i,j] = k1[i]*(x1(j) - cx1[i])**2
        TOP[j] += hist[i,j]

icycle = 1
while icycle < niter:
    for j in range(nbin):
        BOT = np.sum(nb_data * np.exp(beta*(F-V1[:,j])))
        rho[j] = TOP[j] / BOT
        F2 = F2 + rho[j]*np.exp(-beta*V1[:,j])

    converged = True
    F2 = -kbt * np.log(F2)

    diff = np.max(np.abs(F2 - F))
    if diff > tol: converged = False
    print 'round = ', icycle, 'diff = ', diff
    icycle += 1

    if ( fifreq != 0 and icycle % fifreq == 0 ) or ( icycle == niter or converged ):
       open(fff_filename, 'w').write("%8i %s\n" % (icycle, " ".join(["%8.3f" % f for f in F2]))) 
       if icycle == niter or converged: break

    F = F2
    F2 = np.zeros(nwin)

# find maximum rho
jmin = np.argmax(rho)
rhomax = rho[jmin]

# jacobian
#for i in range(nbin):
#    rho[i] = rho[i] / x1(i)**2
#rhomax = np.sum(rho[nbin-5:])/5

print 'maximum density at: x = ', x1(jmin)
jmin = 27
rhomax = rho[jmin]

# make PMF from the rho
np.seterr(divide='ignore')
pmf = -kbt * np.log(rho/rhomax)

open(pmf_filename, 'w').write("\n".join(["%8.3f %12.6f" % (x1(j), pmf[j]) for j in range(nbin)]))
open(rho_filename, 'w').write("\n".join(["%8.3f %12.6f" % (x1(j), rho[j]) for j in range(nbin)])) 
