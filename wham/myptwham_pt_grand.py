from StringIO import StringIO
import sys, os
import numpy as np

os.environ["CC"] = "gcc-4.9"
os.environ["CXX"] = "g++-4.9"

debug = False
n_max = False
if len(sys.argv) > 1: n_max = int(sys.argv[1])

input = sys.stdin
pmf_filename = input.readline().strip() # stores pmf
rho_filename = input.readline().strip() # stores average density
bia_filename = input.readline().strip() # stores biased distribution
fff_filename = input.readline().strip() # stores F(i)
temperature = float(input.readline().strip())

xmin, xmax, deltax, is_x_periodic = map(float, input.readline().strip().split())
umin, umax, deltau, ntemp = map(float, input.readline().strip().split())
vmin, vmax, deltav = map(float, input.readline().strip().split())
nwin, niter, fifreq  = map(int, input.readline().strip().split())
tol = map(float, input.readline().strip().split())
is_x_periodic = bool(is_x_periodic)
nbinx = int((xmax - xmin) / deltax + 0.5)
nbinu = int(abs(umax - umin) / deltau + 0.5)
nbinv = int(abs(vmax - vmin) / deltav + 0.5)
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
hist = np.zeros((nwin, ntemp, nbinx, nbinu, nbinv), dtype=np.int)
nb_data = np.zeros((nwin, ntemp), dtype=np.int)
x1 = lambda j: xmin + (j+1)*deltax - 0.5*deltax
u1 = lambda j: (j+1)*deltau - 0.5*deltau
v1 = lambda j: (j+1)*deltav - 0.5*deltav
energy = np.zeros((nbinx, nbinu))
press = 1.01325 * 1.4383 * 10**-5


data_range = [[None, None], [None, None], [None, None]]

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
            vdata = []
            count = 0
            for line in open(fname):
                time, x, u, v = map(float, line.strip().split()[:4])
                xdata.append(x)
                udata.append(u)
                vdata.append(v)
                if debug and len(xdata) > 10000: break
                if n_max and len(xdata) > n_max: break
            x = np.array(xdata)
            u = np.array(udata)
            v = np.array(vdata)
            u = u - k*(x-cx)**2 #+ press * v
            xbins = [xmin+i*deltax for i in range(nbinx+1)]
            ubins = [umin+i*deltau for i in range(nbinu+1)]
            vbins = [vmin+i*deltav for i in range(nbinv+1)]
            data = np.array((x,u,v)).transpose()

            hist[ihist, jtemp], edges = np.histogramdd(data, bins=(xbins, ubins, vbins), range=((xmin, xmax), (umin, umax), (vmin, vmax)))
            nb_data[ihist, jtemp] = np.sum(hist[ihist,jtemp])

            if data_range[0][0] is None or np.min(x) < data_range[0][0]: data_range[0][0] = np.min(x)
            if data_range[0][1] is None or np.max(x) > data_range[0][1]: data_range[0][1] = np.max(x)
            if data_range[1][0] is None or np.min(u) < data_range[1][0]: data_range[1][0] = np.min(u)
            if data_range[1][1] is None or np.max(u) > data_range[1][1]: data_range[1][1] = np.max(u)
            if data_range[2][0] is None or np.min(v) < data_range[2][0]: data_range[2][0] = np.min(v)
            if data_range[2][1] is None or np.max(v) > data_range[2][1]: data_range[2][1] = np.max(v)
            xedges = edges[0]
            uedges = edges[1]

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
print 'minv:', '%8.3f' % data_range[2][0], 'maxu:', '%8.3f' % data_range[2][1]

print hist.shape

# write biased distribution
f = open(bia_filename, 'w')
for j in range(nbinx):
    for k in range(nbinu):
        f.write("%8d\n" % np.sum(hist[:,:,j,k]))

# iterate wham equation to unbias and recombine the histogram
TOP = np.zeros((nbinx, nbinu, nbinv), dtype=np.int32)
BOT = np.zeros((nbinx, nbinu, nbinv))
W1 = np.zeros((nwin, ntemp, nbinx))
U1 = np.zeros((nwin, ntemp, nbinu))
V1 = np.zeros((nwin, ntemp, nbinv))

for i in range(nwin):
    for j in range(ntemp):
        for k in range(nbinx):
            W1[i,j,k] = k1[i]*(x1(k) - cx1[i])**2

        for l in range(nbinu):
            U1[i,j,l] = u1(l)

        for m in range(nbinv):
            V1[i,j,m] = v1(m) * press

for k in range(nbinx):
    for l in range(nbinu):
        for m in range(nbinv):
            TOP[k,l,m] = np.sum(hist[:,:,k,l,m])


np.set_printoptions(linewidth=200)

from scipy import weave
from scipy.weave import converters

def wham2d(nb_data, TOP, nbinx, nbinu, nbinv, W1, V1, U1, beta, beta0, F=None):
    icycle = 1
    rho = np.zeros((nbinx, nbinu, nbinv), np.double)
    if F is None: F = np.zeros((nwin, ntemp))
    F2 = np.zeros((nwin, ntemp), np.double)

    while icycle < niter:
        code_pragma = """
            double beta1;
            beta1 = beta0;
            #pragma omp parallel num_threads(nthreads)
            {
                #pragma omp for collapse(3) 
                for (int k=0; k<nbinx; k++) {
                    for (int l=0; l<nbinu; l++) {
                        for (int m=0; m<nbinv; m++) {
                            double BOT = 0.0;
                            for (int i=0; i<nwin; i++) {
                                for (int j=0; j<ntemp; j++) {
                                    BOT += nb_data(i,j)*exp(F(i,j)-beta(i,j)*(W1(i,j,k)+U1(i,j,l)+V1(i,j,m)) +beta1*(U1(i,j,l)+V1(i,j,m)));
                                }
                            }

                            if (BOT < 1e-100 || TOP(k,l,m) == 0) continue;
                            rho(k,l,m) = TOP(k,l,m) / BOT;
                        }
                    }
                }

                #pragma omp for collapse(2)
                for (int i=0; i<nwin; i++) {
                    for (int j=0; j<ntemp; j++) {
                        for (int k=0; k<nbinx; k++) {
                            for (int l=0; l<nbinu; l++) {
                                for (int m=0; m<nbinv; m++) {
                                    F2(i,j) += rho(k,l,m)*exp(-beta(i,j)*(W1(i,j,k)+U1(i,j,l)+V1(i,j,m)) + beta1*(U1(i,j,l)+V1(i,j,m))); 
                                }
                            }
                        }
                    }
                }
            }
        """

        nthreads = 4
        weave.inline(code_pragma, ['F', 'F2', 'rho', 'nb_data', 'beta', 'W1', 'U1', 'V1', 'beta0', 'TOP', 'nbinx', 'nbinu', 'nbinv', 'nwin', 'ntemp', 'nthreads'], type_converters=converters.blitz, extra_compile_args=['-O3 -fopenmp'], extra_link_args=['-O3 -fopenmp'], headers=['<omp.h>'])#, library_dirs=['/Users/sunhwan/local/python/lib'])

        converged = True
        F2 = -np.log(F2)
        F2 = F2 -np.min(F2)

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
    F, rho = wham2d(nb_data, TOP, nbinx, nbinu, nbinv, W1, V1, U1, beta, beta0, F)
    np.savetxt(fff, F)

    # jacobian
    for j in range(nbinx):
        rho[j] = rho[j] / x1(j)**2

    # average energy
    avgur = np.zeros(nbinx)
    avgur2 = np.zeros(nbinx)
    avgvr = np.zeros(nbinx)
    rho = rho / np.sum(rho)
    for k in range(nbinx):
        for l in range(nbinu):
            for m in range(nbinv):
                if not (TOP[k,l,m] > 0): continue
                avgur[k] += rho[k,l,m]/np.sum(rho[k]) * u1(l)
                avgur2[k] += rho[k,l,m]/np.sum(rho[k]) * u1(l) * u1(l)
                avgvr[k] += rho[k,l,m]/np.sum(rho[k]) * v1(m)

    # find maximum rho
    rho = np.sum(rho, axis=(1,2))
    jmin = np.argmax(rho)
    rhomax = rho[jmin] 
    #print 'maximum density at: x = ', x1(jmin)

    x0 = int(( 10.55 - xmin ) / deltax)
    rhomax = np.sum(rho[x0-5:x0+5])/10
    avgu = np.sum(avgur[nbinx-10:])/10
    avgv = np.sum(avgvr[nbinx-10:])/10
    #cv = ( avgur2 - avgur**2 ) / kbt / temperature 
    #avgcv = np.average(cv)

    print temperature, avgu, avgv

    # make PMF from the rho
    np.seterr(divide='ignore')
    pmf = -kbt * np.log(rho/rhomax)
    open("%s.%d" % (pmf_filename, i), 'w').write("\n".join(["%8.3f %12.8f %12.8f %12.8f" % (x1(j), pmf[j], avgvr[j]-avgv, avgur[j]-avgu) for j in range(nbinx)]))
    open("%s.%d" % (rho_filename, i), 'w').write("\n".join(["%8.3f %12.8f" % (x1(j), rho[j]) for j in range(nbinx)])) 
