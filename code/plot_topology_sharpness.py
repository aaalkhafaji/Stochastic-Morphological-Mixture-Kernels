"""Render the exact one-pixel sharpness examples from Proposition 5.1."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from morphology import betti
root=Path(__file__).resolve().parents[1]
a=np.zeros((3,3),bool);a[::2,::2]=True
b=a.copy();b[1,1]=True
c=np.ones((5,5),bool);c[2,1:4]=False;c[1:4,2]=False
d=c.copy();d[2,2]=True
fig,axs=plt.subplots(1,4,figsize=(9,2.7))
for ax,x,title,k in zip(axs,[a,b,c,d],['Four components','Center added','One bounded hole','Center filled'],[0,0,1,1]):
 ax.imshow(x,cmap='Greys',vmin=0,vmax=1,interpolation='nearest')
 ax.set_xticks(np.arange(-.5,len(x),1),minor=True);ax.set_yticks(np.arange(-.5,len(x),1),minor=True)
 ax.grid(which='minor',color='#b8c2ca',linewidth=.6)
 ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
 ax.set_title(title,fontsize=10,pad=9)
 ax.set_xlabel(fr'$\beta_{k}={betti(x)[k]}$',fontsize=12,labelpad=8)
 if title in ['Center added','Center filled']:
  mid=len(x)//2;ax.add_patch(Rectangle((mid-.5,mid-.5),1,1,fill=False,edgecolor='#cb4b35',linewidth=2))
fig.tight_layout(pad=1.2)
for suffix in ['pdf','png']:fig.savefig(root/'figures'/f'topology_sharpness.{suffix}',dpi=190,bbox_inches='tight')
