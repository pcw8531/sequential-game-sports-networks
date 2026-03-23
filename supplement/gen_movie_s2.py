"""
SI Movie S2: Four-Scenario Attractor Migration (Fig 4D style)
==============================================================
2x2 grid, all 4 scenarios simultaneously.
Larger figure, centered network insets, bigger fonts.
Output: SI_Movie_S2.gif (pillow, Windows compatible, ≤10 MB)
"""

import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
import math
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

N_ba=50; m_ba=3
G_ba=nx.barabasi_albert_graph(N_ba,m_ba,seed=42)
pos_ba=nx.spring_layout(G_ba,seed=42,k=0.35,iterations=100)
deg_arr=np.array([G_ba.degree(n) for n in G_ba.nodes()])
ce=np.array(list(nx.eigenvector_centrality_numpy(G_ba).values()))
ce=(ce-ce.min())/(ce.max()-ce.min()+1e-9)
k_m=deg_arr.mean(); k_s=deg_arr.std()
hub_set=set(np.where(deg_arr>(k_m+1.5*k_s))[0])
dn=(deg_arr-deg_arr.min())/(deg_arr.max()-deg_arr.min()+1e-9)
nsz=30+180*dn**1.5
ce_rank_desc=np.argsort(-ce); ce_rank_asc=np.argsort(ce)
el=list(G_ba.edges())
nxp=np.array([pos_ba[n][0] for n in G_ba.nodes()])
nyp=np.array([pos_ba[n][1] for n in G_ba.nodes()])

def to_2d(p,f,o): return (f+0.5*o,(math.sqrt(3)/2.0)*o)
def from_2d_to_bary(x2d,y2d):
    z=(2.0/math.sqrt(3))*y2d; y=x2d-0.5*z; x=1.0-(y+z); return(x,y,z)
def compute_flow(h,l,n,ht,lt,nt,steps=5):
    for _ in range(steps):
        dist=math.sqrt((h-ht)**2+(l-lt)**2+(n-nt)**2)
        sc=0.30+0.20*min(dist*2,1.0)
        h+=sc*(ht-h); l+=sc*(lt-l); n+=sc*(nt-n)
        h=max(0,h); l=max(0,l); n=max(0,n)
        s=h+l+n
        if s<1e-12: break
        h/=s; l/=s; n/=s
    return h,l,n

GAMMA_VALS=[0.0,0.1,0.3,0.5,1.0]
SCENARIOS=[
    {'key':'delta','color':'#1A6B3C','label':r'D-$\delta$  full protection',
     'attractors':{0.0:(0.10,0.85,0.05),0.1:(0.30,0.55,0.15),0.3:(0.48,0.28,0.24),0.5:(0.65,0.12,0.23),1.0:(0.85,0.03,0.12)},
     'n_fail':{0.0:45,0.1:35,0.3:25,0.5:12,1.0:0},'fail_who':'periphery'},
    {'key':'alpha','color':'#7B2D8E','label':r'D-$\alpha$  coexistence',
     'attractors':{0.0:(0.10,0.85,0.05),0.1:(0.22,0.62,0.16),0.3:(0.35,0.40,0.25),0.5:(0.42,0.30,0.28),1.0:(0.50,0.20,0.30)},
     'n_fail':{0.0:45,0.1:32,0.3:22,0.5:16,1.0:10},'fail_who':'periphery'},
    {'key':'gamma_sc','color':'#D4780A','label':r'D-$\gamma$  partial coexistence',
     'attractors':{0.0:(0.10,0.85,0.05),0.1:(0.14,0.76,0.10),0.3:(0.20,0.62,0.18),0.5:(0.26,0.52,0.22),1.0:(0.33,0.40,0.27)},
     'n_fail':{0.0:45,0.1:42,0.3:38,0.5:34,1.0:28},'fail_who':'hubs'},
    {'key':'beta','color':'#C0392B','label':r'D-$\beta$  system failure',
     'attractors':{0.0:(0.10,0.85,0.05),0.1:(0.13,0.78,0.09),0.3:(0.16,0.70,0.14),0.5:(0.20,0.63,0.17),1.0:(0.25,0.55,0.20)},
     'n_fail':{0.0:45,0.1:43,0.3:42,0.5:40,1.0:38},'fail_who':'hubs'},
]

for sc in SCENARIOS:
    for gv in sc['attractors']:
        p,f,o=sc['attractors'][gv]; s=p+f+o; sc['attractors'][gv]=(p/s,f/s,o/s)
    pts=[sc['attractors'][gv] for gv in GAMMA_VALS]
    pts_2d=[to_2d(p,f,o) for p,f,o in pts]
    xs=[p[0] for p in pts_2d]; ys=[p[1] for p in pts_2d]
    t_orig=np.array([0,1,2,3,4],dtype=float); t_fine=np.linspace(0,4,200)
    sc['xs_smooth']=CubicSpline(t_orig,xs,bc_type='natural')(t_fine)
    sc['ys_smooth']=CubicSpline(t_orig,ys,bc_type='natural')(t_fine)
    sc['pts_2d']=pts_2d

NGX,NGY=14,14
x2d_q=np.linspace(0.03,0.97,NGX); y2d_q=np.linspace(0.03,math.sqrt(3)/2-0.03,NGY)
XQ,YQ=np.meshgrid(x2d_q,y2d_q)
qmask=np.ones_like(XQ,dtype=bool)
for i in range(NGY):
    for j in range(NGX):
        h,l,n=from_2d_to_bary(XQ[i,j],YQ[i,j])
        if h<-0.01 or l<-0.01 or n<-0.01: qmask[i,j]=False

FLOW_FIELDS={}  # kept empty, flow computed dynamically per frame

def interp_attractor_sc(sc, gamma_val):
    """Interpolate a scenario's attractor at any gamma value."""
    gkeys = sorted(sc['attractors'].keys())
    if gamma_val <= gkeys[0]: return sc['attractors'][gkeys[0]]
    if gamma_val >= gkeys[-1]: return sc['attractors'][gkeys[-1]]
    for i in range(len(gkeys)-1):
        if gkeys[i] <= gamma_val <= gkeys[i+1]:
            t = (gamma_val - gkeys[i]) / (gkeys[i+1] - gkeys[i])
            a0 = sc['attractors'][gkeys[i]]; a1 = sc['attractors'][gkeys[i+1]]
            p = a0[0]+t*(a1[0]-a0[0]); f = a0[1]+t*(a1[1]-a0[1]); o = a0[2]+t*(a1[2]-a0[2])
            s = p+f+o; return (p/s, f/s, o/s)
    return sc['attractors'][gkeys[-1]]

def compute_flow_field(sc, gamma_val):
    """Compute quiver field pointing toward the current attractor."""
    ht, lt, nt = interp_attractor_sc(sc, gamma_val)
    UQ = np.full_like(XQ, np.nan); VQ = np.full_like(XQ, np.nan)
    for i in range(NGY):
        for j in range(NGX):
            if not qmask[i,j]: continue
            h,l,n = from_2d_to_bary(XQ[i,j], YQ[i,j])
            h=max(0,h); l=max(0,l); n=max(0,n)
            ss=h+l+n
            if ss<1e-12: continue
            h/=ss; l/=ss; n/=ss
            h2,l2,n2 = compute_flow(h,l,n,ht,lt,nt)
            xo,yo = to_2d(h,l,n); xn,yn = to_2d(h2,l2,n2)
            UQ[i,j]=xn-xo; VQ[i,j]=yn-yo
    return UQ, VQ

v_left=np.array([0,0]); v_right=np.array([1,0]); v_top=np.array([0.5,math.sqrt(3)/2])

###############################################################################
# Drawing helpers — LARGER FONTS
###############################################################################
def draw_gridlines(ax):
    for frac in [0.2,0.4,0.6,0.8]:
        x0=0.5*frac; y0=(math.sqrt(3)/2)*frac; x1=1.0-0.5*frac
        ax.plot([x0,x1],[y0,y0],color='gray',alpha=0.22,lw=0.5,zorder=1)
        bx=frac; tx=0.5+0.5*(1-frac); ty=(math.sqrt(3)/2)*(1-frac)
        ax.plot([bx,tx],[0,ty],color='gray',alpha=0.22,lw=0.5,zorder=1)
        ax.plot([frac*0.5,frac+(1-frac)*0.5],[frac*math.sqrt(3)/2,0],color='gray',alpha=0.22,lw=0.5,zorder=1)

def draw_axis_labels(ax):
    lfs = 13  # axis label font
    tfs = 11  # tick font
    tick_off = 0.055

    mid_bot=(v_left+v_right)/2
    ax.text(mid_bot[0],mid_bot[1]-0.07,'Failure cascade',
            fontsize=lfs,ha='center',va='top',color='#8B0000',fontweight='bold')

    mid_left=0.58*v_left+0.42*v_top
    px=-math.sin(math.radians(60)); py=math.cos(math.radians(60))
    ax.text(mid_left[0]+0.13*px,mid_left[1]+0.13*py,'Protection level ($f_p$)',
            fontsize=lfs,ha='center',va='center',color='#2874A6',fontweight='bold',rotation=60)

    mid_right=0.58*v_right+0.42*v_top
    rx=math.sin(math.radians(60)); ry=math.cos(math.radians(60))
    ax.text(mid_right[0]+0.13*rx,mid_right[1]+0.13*ry,'Sequential obs. ($O_i$)',
            fontsize=lfs,ha='center',va='center',color='#555',fontweight='bold',rotation=-60)

    for val in [0.0,1.0]:
        tx=v_left[0]+val*(v_right[0]-v_left[0])
        ax.text(tx,-tick_off,f'{val:.1f}',fontsize=tfs,ha='center',va='top',color='#555')
    for val in [0.0,1.0]:
        pt=v_left+val*(v_top-v_left)
        dx=-tick_off*math.cos(math.radians(30)); dy=-tick_off*math.sin(math.radians(30))
        ax.text(pt[0]+dx,pt[1]+dy,f'{val:.1f}',fontsize=tfs,ha='right',va='center',color='#555')
    for val in [0.0,1.0]:
        pt=v_right+val*(v_top-v_right)
        dx=tick_off*math.cos(math.radians(30)); dy=-tick_off*math.sin(math.radians(30))
        ax.text(pt[0]+dx,pt[1]+dy,f'{val:.1f}',fontsize=tfs,ha='left',va='center',color='#555')

def draw_network_centered(iax, sc_cfg, gamma_val):
    iax.clear(); iax.axis('off'); iax.patch.set_alpha(0.0)
    pad=0.08
    iax.set_xlim(nxp.min()-pad,nxp.max()+pad)
    iax.set_ylim(nyp.min()-pad,nyp.max()+pad)
    iax.set_aspect('equal')

    gkeys=sorted(sc_cfg['n_fail'].keys()); nf=sc_cfg['n_fail'][gkeys[-1]]
    for i in range(len(gkeys)-1):
        if gkeys[i]<=gamma_val<=gkeys[i+1]:
            t=(gamma_val-gkeys[i])/(gkeys[i+1]-gkeys[i])
            nf=int(round(sc_cfg['n_fail'][gkeys[i]]*(1-t)+sc_cfg['n_fail'][gkeys[i+1]]*t)); break
    nf=max(0,min(nf,N_ba))
    if nf==0: fail_nodes=set()
    elif sc_cfg['fail_who']=='periphery': fail_nodes=set(ce_rank_asc[:nf])
    else: fail_nodes=set(ce_rank_desc[:nf])

    segs,ecols,ewids=[],[],[]
    for u,v in el:
        wn=(deg_arr[u]+deg_arr[v])/(2.0*deg_arr.max()); uf=u in fail_nodes; vf=v in fail_nodes
        if not uf and not vf:
            c=plt.cm.Blues(0.25+0.65*wn); segs.append([pos_ba[u],pos_ba[v]]); ecols.append((*c[:3],0.12+0.45*wn)); ewids.append(0.5+1.5*wn)
        elif uf and vf:
            segs.append([pos_ba[u],pos_ba[v]]); ecols.append((0.75,0.20,0.20,0.05+0.12*wn)); ewids.append(0.2+0.6*wn)
        else:
            segs.append([pos_ba[u],pos_ba[v]]); ecols.append((0.55,0.55,0.55,0.03+0.06*wn)); ewids.append(0.15+0.25*wn)
    iax.add_collection(LineCollection(segs,colors=ecols,linewidths=ewids,zorder=2))

    accent=sc_cfg['color']
    for i in range(N_ba):
        x,y=pos_ba[i]; ih=i in hub_set
        if i in fail_nodes: nc=(0.60+0.35*ce[i],0.10+0.08*(1-ce[i]),0.10+0.08*(1-ce[i])); an=0.65
        else: val=np.clip(ce[i],0.1,1.0); nc=(0.10+0.25*(1-val),0.15+0.30*(1-val),0.50+0.50*val); an=0.75
        iax.scatter(x,y,s=nsz[i]*0.5,c=[nc],alpha=an,
                    edgecolors='#222' if ih else '#888',linewidths=0.4 if ih else 0.08,
                    zorder=4 if ih else 3)
    for n in hub_set:
        x,y=pos_ba[n]; rc=accent if n not in fail_nodes else '#8B0000'
        iax.scatter(x,y,s=nsz[n]*2.5,facecolors='none',edgecolors=rc,linewidths=0.5,alpha=0.35,zorder=5)

###############################################################################
# Figure: 2x2 — LARGER SIZE
###############################################################################
N_FRAMES = 50

fig, axes = plt.subplots(2, 2, figsize=(18, 15), dpi=100)
fig.subplots_adjust(hspace=0.12, wspace=0.18, top=0.91, bottom=0.03, left=0.03, right=0.97)
axes_flat = axes.flatten()

# Network insets CENTERED within each subplot
# These are in figure coordinates — computed from subplot positions
# Top row: y ~ 0.55-0.94, Bottom row: y ~ 0.03-0.47
# Left col: x ~ 0.03-0.48, Right col: x ~ 0.52-0.97
inset_w = 0.18; inset_h = 0.18

inset_positions = [
    [0.03 + (0.45-inset_w)/2,  0.51 + (0.39-inset_h)/2 - 0.04,  inset_w, inset_h],  # top-left centered
    [0.52 + (0.45-inset_w)/2,  0.51 + (0.39-inset_h)/2 - 0.04,  inset_w, inset_h],  # top-right centered
    [0.03 + (0.45-inset_w)/2,  0.03 + (0.39-inset_h)/2 + 0.01,  inset_w, inset_h],  # bottom-left centered
    [0.52 + (0.45-inset_w)/2,  0.03 + (0.39-inset_h)/2 + 0.01,  inset_w, inset_h],  # bottom-right centered
]
inset_axes = [fig.add_axes(pos) for pos in inset_positions]

def animate(fi):
    progress = fi / (N_FRAMES - 1)
    gamma_now = progress * 1.0

    for si, sc in enumerate(SCENARIOS):
        ax = axes_flat[si]
        iax = inset_axes[si]
        sc_c = sc['color']

        ax.clear()
        ax.set_aspect('equal','box')
        ax.set_xlim(-0.20,1.20)
        ax.set_ylim(-0.18,math.sqrt(3)/2+0.20)
        ax.axis('off')

        draw_gridlines(ax)
        draw_axis_labels(ax)

        # Dynamic flow field — arrows follow current attractor position
        UQ, VQ = compute_flow_field(sc, gamma_now)
        ax.quiver(XQ,YQ,UQ,VQ,color='#AAAAAA',alpha=0.30,
                  scale=3.5,width=0.004,headwidth=5,headlength=5,zorder=3)

        # Trajectory
        n_pts=max(1,int(progress*199)+1)
        xs_d=sc['xs_smooth'][:n_pts]; ys_d=sc['ys_smooth'][:n_pts]

        if len(xs_d)>1:
            ax.plot(xs_d,ys_d,'-',color=sc_c,linewidth=3.5,alpha=0.85,zorder=10,
                    path_effects=[pe.Stroke(linewidth=5.5,foreground='white',alpha=0.6),pe.Normal()])

        # Direction arrows
        for fa in [0.30,0.65]:
            ia=min(int(fa*n_pts),n_pts-1); ib=max(ia-4,0); ic=min(ia+4,n_pts-1)
            if ic>ib and n_pts>10:
                ax.annotate('',xy=(xs_d[ic],ys_d[ic]),xytext=(xs_d[ib],ys_d[ib]),
                           arrowprops=dict(arrowstyle='->',color=sc_c,lw=2.5,mutation_scale=16),zorder=12)

        # gamma=0 start
        sx,sy=sc['pts_2d'][0]
        ax.plot(sx,sy,'o',color='#333',markersize=10,markeredgecolor='black',markeredgewidth=1.2,zorder=18)
        ax.text(sx-0.08,sy-0.02,r'$\gamma=0$'+'\n(simultaneous)',
                fontsize=10,ha='left',va='top',color='#333',fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=20)

        # Waypoints
        for wi in range(1,5):
            gv=GAMMA_VALS[wi]; wp=wi/4.0
            if progress>=wp:
                wx,wy=sc['pts_2d'][wi]
                if wi<4:
                    ax.plot(wx,wy,'o',color=sc_c,markersize=7,
                            markeredgecolor='white',markeredgewidth=0.8,zorder=18)
                    ax.text(wx+0.03,wy+0.02,f'$\\gamma$={gv}',fontsize=10,
                            ha='left',va='bottom',color=sc_c,fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=20)
                else:
                    ax.plot(wx,wy,'o',color=sc_c,markersize=24,alpha=0.15,zorder=14)
                    ax.plot(wx,wy,'o',color=sc_c,markersize=12,
                            markeredgecolor='black',markeredgewidth=1.8,zorder=19)
                    ax.text(wx,wy+0.04,r'$\gamma=1.0$',fontsize=11,
                            ha='center',va='bottom',color=sc_c,fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=20)

        # Moving dot
        if n_pts>0:
            ax.plot(xs_d[-1],ys_d[-1],'o',color=sc_c,markersize=9,
                    markeredgecolor='white',markeredgewidth=1.5,zorder=17)

        # Panel title
        ax.text(0.50,0.98,sc['label'],transform=ax.transAxes,
                fontsize=16,ha='center',va='bottom',color=sc_c,fontweight='bold')

        # Network
        draw_network_centered(iax, sc, gamma_now)

    # Overall title
    fig.suptitle(f'SI Movie S2: Four-Scenario Attractor Migration   |   $\\gamma$ = {gamma_now:.2f}',
                 fontsize=18, fontweight='bold', y=0.96, color='#333')

print(f"Generating SI Movie S2 ({N_FRAMES} frames, 2x2 grid, 18x16 figure)...")
ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=250, blit=False)
ani.save('SI_Movie_S2.gif', writer='pillow', fps=4, dpi=100)
plt.close(fig)

fsize = os.path.getsize('SI_Movie_S2.gif')
print(f"Saved: SI_Movie_S2.gif ({fsize/1024/1024:.2f} MB)")

try:
    from IPython.display import Image, display
    display(Image(filename='SI_Movie_S2.gif'))
except: pass

# === MP4 for PNAS submission ===
ani.save('SI_Movie_S2.mp4', writer='ffmpeg', fps=4, dpi=100,
         bitrate=1800, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f"Saved: SI_Movie_S2.mp4 ({os.path.getsize('SI_Movie_S2.mp4')/1024/1024:.2f} MB)")
