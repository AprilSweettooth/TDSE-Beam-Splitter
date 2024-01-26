# assume $\bar{h}=1$
import numpy as np
import scipy.sparse
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def norm(psi, dx):
    norm = np.sum(np.square(np.abs(psi)))*dx
    return psi/np.sqrt(norm)

def wave_packet(x,x0,dx,mom=0,sigma=0.2):
    # A = (2 * np.pi * sigma**2)**(-0.25)
    return norm(np.exp(1j*mom*x)*np.exp(-np.square(x-x0)/(4*sigma**2),dtype=complex),dx)
    # return (np.pi * sigma**2)**(-0.25)*np.exp(1j*mom*x)*np.exp(-np.square(x-x0)/(2*sigma**2),dtype=complex)

def gaussian_wavepacket(x, x0, sigma0, p0):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)

def complex_plot(x,y,ab=True,**kwargs):
    real = np.real(y)
    imag = np.imag(y)
    if ab:
        a,*_ = plt.plot(x,real,label='Re',color='red',**kwargs)
        b,*_ = plt.plot(x,imag,label='Im',color='blue',**kwargs)
        plt.xlim(-2,2)
        plt.legend()
        return
    else:
        p,*_ = plt.plot(x,np.abs(y),color='black',label='$\sqrt{P}$')
        plt.xlim(-2,2)
    return p

def d_dxdx(psi,dx,h=1,m=100):
    dphi_dxdx = -2*psi
    dphi_dxdx[:-1] += psi[1:]
    dphi_dxdx[1:] += psi[:-1]
    return h/(2*m)*dphi_dxdx/dx**2

def d_dt(psi,dx,h=1,m=100,V=0):
    """multiply Schrodinger's eqn by -i"""
    return 1j * d_dxdx(psi,dx,h,m) - 1j*V*psi/h

def euler(psi, dt, dx, **kwargs):
    return psi + dt * d_dt(psi, dx, **kwargs)

def rk4(psi, dt, dx, **kwargs):
    k1 = d_dt(psi,dx, **kwargs)
    k2 = d_dt(psi+dt/2*k1,dx, **kwargs)
    k3 = d_dt(psi+dt/2*k2,dx, **kwargs)
    k4 = d_dt(psi+dt*k3,dx, **kwargs)
    return psi + dt/6*(k1+2*k2+2*k3+k4)

#Hamiltonian evolution
def evolution(psi0,
            m,
            V,
            dx,
            steps=100000, 
            dt=0.1, 
            # normalize=True,
            save_every=1000):
    simulation_steps = [np.copy(psi0)]
    # U = time_evolution_operator(H, dt)
    psi = psi0
    overlap = psi0
    for t in range(steps):
        psi = rk4(psi,dt,dx,m=m,V=V)
        # if normalize:
        # psi = norm(psi, dx)
        if save_every is not None and (t+1) % save_every == 0:
            # check norm
            # print(np.sum(np.square(np.abs(psi)))*dx)
            # check overlap
            # print(abs(np.sum([np.conjugate(i)*j for i,j in zip(overlap, psi)])*dx))
            simulation_steps.append(np.copy(psi))
    return simulation_steps

def box_init():
    # We paint the walls of the double slit with rectangles.
    wall_left = Rectangle((-3,-3), 1, 2.5,  color="grey") # (x0, y0), width, height
    wall_right= Rectangle((2,-3), 1,2.5,     color="grey")

    # We add the rectangular patches to the plot.
    plt.gca().add_patch(wall_left)
    plt.gca().add_patch(wall_right)
    plt.xlim(-3,3)
    plt.ylim(-0.5,2)


def barrier_init():
    barrier= Rectangle((1.4,-3), 0.2,2.5,     color="grey")
    plt.gca().add_patch(barrier)
    plt.xlim(-2,4)
    plt.ylim(-0.5,2)

def harmonic_init(x,y):
    plt.fill_between(x,y,-0.5,color='grey')
    plt.xlim(-10,10)
    plt.ylim(-0.5,2)

def beam_init(x,y):
    plt.fill_between(x,y,-0.5,color='grey')
    plt.xlim(-10,10)
    plt.ylim(-0.5,2)

def radar_init(x,y):
    plt.fill_between(x,y,-0.5,color='grey')
    plt.xlim(-10,10)
    plt.ylim(-0.5,2)

def animate(x,simulation_steps,init_func=None,name='free',V_sho=None):
    # fig = plt.figure()
    fig, ax = plt.subplots()
    prob = complex_plot(x,simulation_steps[0],ab=False)
    plt.xlim(-10,10)
    plt.ylim(-0.5,2)
    if init_func:
        if name=='harmonic' or name=='beam':
            init_func(x,V_sho)
        else:
            init_func()
    # plt.legend()

    def animate(frame):
        prob.set_data((x, np.abs(simulation_steps[frame])))
        # re.set_data((x, np.real(simulation_steps[frame])))
        # im.set_data((x, np.imag(simulation_steps[frame])))
        return prob,


    anim = FuncAnimation(fig, animate, frames=int(len(simulation_steps)), interval=50)
    anim.save(name+'.gif', writer='Pillow')
    plt.close()
    return anim

def QFI(sims, delta_d):
    qfi = []
    for i in range(len(sims[0])):
        d_psi_dd = (sims[0][i] - sims[1][i])/delta_d
        qfi.append(4*np.real(np.sum(np.square(np.abs(d_psi_dd)))*delta_d - delta_d**2*abs(d_psi_dd@np.conj(sims[0][i].T))**2))
    return qfi

def plot_qfi(t,qfi):
    fig, ax = plt.subplots()
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 1)
    graph, = plt.plot([], [], '-')

    def init():
        return graph,

    def animate(i):
        graph.set_data(t[i],qfi[i])
        return graph,

    ani = FuncAnimation(fig, animate, frames=range(len(t)), interval=50, save_count=len(t),
                        init_func=init, blit=True)
    ani.save('qfi.gif', writer='Pillow')

def two_plot(t,qfi,x,simulation_steps,V):
    fig, (axl, axr) = plt.subplots(
        ncols=2,
        # sharey=True,
        figsize=(12, 4),
    )
    # axl.set_aspect(2)
    axl.set_box_aspect(1 / 1)
    axr.set_box_aspect(1 / 3)
    # axr.yaxis.set_visible(False)
    # axr.xaxis.set_ticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])

    # draw circle with initial point in left Axes
    point, = axl.plot([], [], '-')
    # point, = axl.plot(t[0], qfi[0], "o")
    axl.set_xlim(0,t[-1])
    axl.set_ylim(min(qfi),max(qfi))
    prob = complex_plot(x,simulation_steps[0],ab=False)
    axr.set_xlim(-10,10)
    axr.set_ylim(-0.5,2)
    axr.fill_between(x,V,-0.5,color='grey')

    def animate(frame):
        prob.set_data((x, np.abs(simulation_steps[frame])))
        point.set_data(t[:frame], qfi[:frame])
        # con.xy1 = x, y
        # con.xy2 = i, y
        # return point, sine, con
        return point, prob


    anim = FuncAnimation(fig, animate, frames=int(len(simulation_steps)), interval=50)
    anim.save('qfi.gif', writer='Pillow')