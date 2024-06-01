import jax.numpy as np
import matplotlib.pyplot as plt


def plot(vi, X, 
    target='vanishing', 
    n=1000, scale=1.5, x_max=1.0, y_max=1.0,
    z_func=lambda x_, y_: 0.0,
    show=False, splitshow=False):

    nvars = X.shape[-1]
    if nvars == 2:
        _plot2d(vi, X, target=target, 
                n=n, scale=scale, x_max=x_max, y_max=y_max,
                show=show, splitshow=splitshow)
    elif nvars == 3:
        _plot3d(vi, X, z_func, target=target, 
        n=n, scale=scale, x_max=x_max, y_max=y_max,
        show=show, splitshow=splitshow)
    else: 
        print(f'Cannot plot {nvars}-variate polynomials')

def _plot2d(vi, X, target='vanishing', n=1000, scale=1.5, x_max=1.0, y_max=1.0, show=False, splitshow=False):

    ## set plot range
    m = np.mean(X, axis=0)
    x_max = y_max = np.max(np.abs(X))
    # x = np.arange(-scale*x_max, scale*x_max, resolution)
    # y = np.arange(-scale*y_max, scale*y_max, resolution)
    x = np.linspace(-scale*x_max, scale*x_max, 50)
    y = np.linspace(-scale*y_max, scale*y_max, 50)
    Z1, Z2 = np.meshgrid(x, y)

    ## set plot setting
    npolys = 0
    if target == 'vanishing':
        # npolys = sum([Gt.shape[-1] for Gt in vi.basis.vanishings()])
        npolys = sum([Bt.n_vanishings() for Bt in vi.basis])
        # npolys = sum([len(Gt) for Gt in vi.basis.vanishings()])
    elif target == 'nonvanishing':
        npolys = sum([Bt.n_nonvanishings() for Bt in vi.basis])

    colors = plt.cm.Dark2(np.linspace(0,1,8))
    linestyles = ['solid','dashed','dashdot', 'dotted']
    nfigs = min(npolys, n)

    for i in range(nfigs):
        f = lambda x_, y_: vi.evaluate(np.array([[x_,y_]]), target=target)[0,i]
        f = np.vectorize(f)
        plt.contour(Z1,Z2,f(Z1, Z2), levels=[0], colors=[colors[i%len(colors)]], linewidths=[1.], linestyles=[linestyles[i%4]])

        if splitshow:
            plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

    if not splitshow:
        plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
        plt.gca().set_aspect('equal', adjustable='box')      
#         plt.savefig('graph_Z.pdf') 
    
    if not splitshow and show: 
        plt.show()


def _plot3d(vi, X, z_func, target='vanishing', n=1000, scale=1.5, x_max=1.0, y_max=1.0, show=False, splitshow=False):

    ## set plot range
    m = np.mean(X, axis=0)
    x_max = y_max = np.max(np.abs(X))
    x = np.linspace(-scale*x_max, scale*x_max, 50)
    y = np.linspace(-scale*y_max, scale*y_max, 50)
    Z1, Z2 = np.meshgrid(x, y)

    ## set plot setting
    npolys = 0
    if target == 'vanishing':
        npolys = sum([np.asarray(Gt).shape[-1] for Gt in vi.basis.vanishings()])
        # npolys = sum([len(Gt) for Gt in vi.basis.vanishings()])
    elif target == 'nonvanishing':
        npolys = sum([np.asarray(Ft).shape[-1] for Ft in vi.basis.nonvanishings()])
    else:
        print('unknown target: %s' % target)

    colors = plt.cm.Dark2(np.linspace(0,1,8))
    linestyles = ['solid','dashed','dashdot', 'dotted']
    nfigs = min(npolys, n)

    for i in range(nfigs):
        f = lambda x_, y_: vi.evaluate(np.array([[x_,y_, z_func(x_,y_)]]), target=target)[0,i]
        f = np.vectorize(f)
        plt.contour(Z1,Z2,f(Z1, Z2), levels=[0], colors=[colors[i%len(colors)]], linewidths=[1.], linestyles=[linestyles[i%4]])

        if splitshow:
            plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

    if not splitshow:
        plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
        plt.gca().set_aspect('equal', adjustable='box')      
#         plt.savefig('graph_Z.pdf') 
    
    if not splitshow and show: 
        plt.show()
