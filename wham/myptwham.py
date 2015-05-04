from StringIO import StringIO
import sys
import os
import numpy as np

debug = False
if len(sys.argv) > 1: debug = True

input = sys.stdin
pmf_filename = input.readline().strip() # stores pmf
rho_filename = input.readline().strip() # stores average density
bia_filename = input.readline().strip() # stores biased distribution
fff_filename = input.readline().strip() # stores F(i)
temperature = float(input.readline().strip())

xmin, xmax, deltax, is_x_periodic = map(float, input.readline().strip().split())
umin, umax, deltau, ntemp = map(float, input.readline().strip().split())
nwin, niter, fifreq  = map(int, input.readline().strip().split())
tol = map(float, input.readline().strip().split())
is_x_periodic = bool(is_x_periodic)
nbinx = int((xmax - xmin) / deltax + 0.5)
nbinu = int(abs(umax - umin) / deltau + 0.5)
ntemp = int(ntemp)
kb = 0.0019872
kbt = kb * temperature
beta0 = 1.0/kbt

if debug:
    temperature = 283.15
    kbt = kb * temperature
    beta0 = 1.0/kbt

k1 = np.zeros(nwin)
cx1 = np.zeros(nwin)
temp = np.zeros(ntemp)
beta = np.zeros((nwin, ntemp))
tseries = np.empty(nwin, dtype='S')
hist = np.zeros((nwin, ntemp, nbinx, nbinu), dtype=np.int)
hist0 = np.zeros((nwin, ntemp, nbinx, nbinu), dtype=np.int)
nb_data = np.zeros((nwin, ntemp, nbinu), dtype=np.int)
x1 = lambda j: xmin + (j+1)*deltax - 0.5*deltax
u1 = lambda j: (j+1)*deltau - 0.5*deltau
energy = np.zeros((nbinx, nbinu))

data_range = [[None, None], [None, None]]

for j in range(ntemp):
    for i in range(nwin):
        fname = input.readline().strip()
        tseries[i] = fname
        line = input.readline().strip()
        cx1[i], k1[i], temp[j] = map(float, line.split()[:3])
        beta[i,j] = 1 / (kb * temp[j])

        def mkhist(fname, xmin, xmax, ymin, ymax, deltax, deltay, ihist, jtemp, k, cx):
            xdata = []
            udata = []
            count = 0
            for line in open(fname):
                time, x, u = map(float, line.strip().split()[:3])
                count += 1
                #if count < 500: continue
                xdata.append(x)
                udata.append(u)
                if debug and len(xdata) > 10000: break
            x = np.array(xdata)
            u = np.array(udata)
            u = u - k*(x-cx)**2
            xbins = [xmin+i*deltax for i in range(nbinx+1)]
            ubins = [umin+i*deltau for i in range(nbinu+1)]

            hist[ihist, jtemp], xedges, uedges = np.histogram2d(x, u, bins=(xbins, ubins), range=((xmin, xmax), (umin, umax)))
            nb_data[ihist, jtemp] = [np.sum(hist[ihist,jtemp,:,i]) for i in range(nbinu)]

            if data_range[0][0] is None or np.min(x) < data_range[0][0]: data_range[0][0] = np.min(x)
            if data_range[0][1] is None or np.max(x) > data_range[0][1]: data_range[0][1] = np.max(x)
            if data_range[1][0] is None or np.min(u) < data_range[1][0]: data_range[1][0] = np.min(u)
            if data_range[1][1] is None or np.max(u) > data_range[1][1]: data_range[1][1] = np.max(u)

            print 'statistics for timeseries # ', ihist
            print 'minx:', '%8.3f' % np.min(x), 'maxx:', '%8.3f' % np.max(x)
            print 'average x', '%8.3f' % np.average(x), 'rms x', '%8.3f' % np.std(x)
            print 'minu:', '%8.3f' % np.min(u), 'maxu:', '%8.3f' % np.max(u)
            print 'average u', '%8.3f' % np.average(u), 'rms u', '%8.3f' % np.std(u)
            print 'statistics for histogram # ', ihist
            print int(np.sum(hist[ihist,jtemp])), 'points in the histogram x'
            print 'average x', '%8.3f' % (np.sum([hist[ihist,jtemp,i,:]*(xedges[i]+xedges[i+1])/2 for i in range(nbinx)])/np.sum(hist[ihist,jtemp]))
            print 'average u', '%8.3f' % (np.sum([hist[ihist,jtemp,:,i]*(uedges[i]+uedges[i+1])/2 for i in range(nbinu)])/np.sum(hist[ihist,jtemp]))
            print

        mkhist(fname, xmin, xmax, umin, umax, deltax, deltau, i, j, k1[i], cx1[i])

print 'minx:', '%8.3f' % data_range[0][0], 'maxx:', '%8.3f' % data_range[0][1]
print 'minu:', '%8.3f' % data_range[1][0], 'maxu:', '%8.3f' % data_range[1][1]

# write biased distribution
f = open(bia_filename, 'w')
for j in range(nbinx):
    for k in range(nbinu):
        f.write("%8d\n" % np.sum(hist[:,:,j,k]))

# iterate wham equation to unbias and recombine the histogram
TOP = np.zeros((nbinx, nbinu), dtype=np.int32)
BOT = np.zeros((nbinx, nbinu))
V1 = np.zeros((nwin, ntemp, nbinx))
U1 = np.zeros((nwin, ntemp, nbinu))
for i in range(nwin):
    for j in range(ntemp):
        for k in range(nbinx):
            for l in range(nbinu):
                V1[i,j,k] = k1[i]*(x1(k) - cx1[i])**2
                U1[i,j,l] = u1(l)
                TOP[k,l] += hist[i,j,k,l]

np.set_printoptions(linewidth=200)

def wham2d(nb_data, TOP, nbinx, nbinu, V1, U1, beta, beta0, F=None):
    icycle = 1
    rho = np.zeros((nbinx, nbinu))
    if F is None: F = np.zeros((nwin, ntemp))
    F2 = np.zeros((nwin, ntemp))
    while icycle < niter:
        for k in range(nbinx):
            for l in range(nbinu):
                BOT = np.sum(np.sum(nb_data, axis=2) * np.exp(F - beta*(V1[:,:,k] + U1[:,:,l]) + beta0*U1[:,:,l]))
                #BOT = np.sum(np.sum(nb_data, axis=2) * np.exp(F - beta*(V1[:,:,k] + U1[:,:,l])))
                if BOT < 1e-100 or TOP[k,l] == 0: continue
                rho[k,l] = TOP[k,l] / BOT
                F2 = F2 + rho[k,l]*np.exp(-beta*(V1[:,:,k] + U1[:,:,l]) + beta0*U1[:,:,l])
                #F2 = F2 + rho[k,l]*np.exp(-beta*(V1[:,:,k] + U1[:,:,l]))

        converged = True
        F2 = -np.log(F2)
        F2 = F2 -np.min(F2)
        #sys.exit()

        diff = np.max(np.abs(F2 - F))

        if diff > tol: converged = False
        print 'round = ', icycle, 'diff = ', diff
        icycle += 1

        if ( fifreq != 0 and icycle % fifreq == 0 ) or ( icycle == niter or converged ):
           print F2
           #open(fff_filename, 'w').write("%8i %s\n" % (icycle, " ".join(["%8.3f" % f for f in F2]))) 
           if icycle == niter or converged: break

        F = F2
        F2 = np.zeros((nwin, ntemp))

    return F2, rho

F = np.zeros((nwin, ntemp))
for i in range(ntemp):
    temperature = temp[i]
    kbt = kb * temperature
    beta0 = 1.0/kbt

    fff = "%s.%d" % (fff_filename, i)
    if i == 0 and os.path.exists(fff):
        F = np.loadtxt(fff)
    F, rho = wham2d(nb_data, TOP, nbinx, nbinu, V1, U1, beta, beta0, F)
    np.savetxt(fff, F)

    # jacobian
    #for j in range(nbinx):
    #    rho[j] = rho[j] / x1(j)**2

    # average energy
    avgur = np.zeros(nbinx)
    avgur2 = np.zeros(nbinx)
    rho = rho / np.sum(rho)
    for k in range(nbinx):
        for l in range(nbinu):
            if not (TOP[k,l] > 0): continue
            avgur[k] += rho[k,l]/np.sum(rho[k,:]) * u1(l) 
            avgur2[k] += rho[k,l]/np.sum(rho[k,:]) * u1(l) * u1(l) 

    # find maximum rho
    rho = np.sum(rho, axis=1)
    jmin = np.argmax(rho)
    rhomax = rho[jmin] 
    #print 'maximum density at: x = ', x1(jmin)

    rhomax = np.sum(rho[nbinx-5:])/5
    avgu = np.sum(avgur[nbinx-5:])/5
    cv = ( avgur2 - avgur ) / kbt / temperature 
    avgcv = 0 #np.average(cv[-5:])

    print temperature, avgu

    # make PMF from the rho
    np.seterr(divide='ignore')
    pmf = -kbt * np.log(rho/rhomax)
    open("%s.%d" % (pmf_filename, i), 'w').write("\n".join(["%8.3f %12.3f %12.3f %12.3f %12.3f" % (x1(j), pmf[j], avgur[j], avgur[j]-avgu, cv[j]-avgcv) for j in range(nbinx)]))
    open("%s.%d" % (rho_filename, i), 'w').write("\n".join(["%8.3f %12.3f" % (x1(j), rho[j]) for j in range(nbinx)])) 
