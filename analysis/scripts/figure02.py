"""Update regional visualization, retaining every audited case and confidence interval."""
from pathlib import Path
import os,json,hashlib,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'figure_data/Figure2.json'
d=json.loads(SOURCE.read_text()); regions=d['regions']
BLUE,TEAL,INK,GRID='#285A80','#32857B','#263C4A','#E7EBEE'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
                     'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
                     'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
                     'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
fig=plt.figure(figsize=(7.2,6.0),facecolor='white')
a=fig.add_axes([.09,.555,.87,.365])
b=fig.add_axes([.09,.115,.365,.30])
c=fig.add_axes([.585,.115,.35,.30])
cax=fig.add_axes([.946,.115,.012,.30])
rng=np.random.default_rng(20260916)
yy=np.linspace(0,1,401)
count=0
for i,r in enumerate(d['pooled']):
    values=np.array([x['dice'] for x in r['positive_cases']]); assert len(values)==r['n_positive']
    kde=gaussian_kde(values,bw_method=.23)
    density=kde(yy)+kde(-yy)+kde(2-yy); density=.34*density/density.max()
    a.fill_betweenx(yy,i-.06-density,i-.06,color=BLUE,alpha=.16,lw=0)
    a.plot(i-.06-density,yy,color=BLUE,lw=.75,alpha=.6)
    a.scatter(i+rng.uniform(.045,.355,len(values)),values,s=8.2,c=BLUE,alpha=.38,lw=0,zorder=3)
    q1,med,q3=np.quantile(values,[.25,.5,.75])
    a.plot([i-.06]*2,[q1,q3],color=BLUE,lw=3.4,solid_capstyle='round')
    a.plot(i-.06,med,'o',ms=4,mfc='white',mec=BLUE,mew=.9)
    count+=len(values)
assert count==1076
a.set(xlim=(-.58,5.55),ylim=(-.035,1.035),ylabel='Dice',
      xticks=range(6),xticklabels=[f'{r["region"]}\n(n = {r["n_positive"]})' for r in d['pooled']],
      yticks=[0,.25,.5,.75,1])
a.tick_params(axis='x',length=0,pad=7)
a.yaxis.grid(True,color=GRID,lw=.65);a.set_axisbelow(True)
a.spines['bottom'].set_visible(False);a.spines['left'].set_bounds(0,1)
a.text(0,1.09,'A',weight='bold',fontsize=12,transform=a.transAxes)
a.text(.06,1.09,'Reference-positive cases',fontsize=10.5,transform=a.transAxes)

for i,r in enumerate(d['pooled']):
    for offset,key,color,marker in [(-.13,'all',BLUE,'o'),(.13,'positive',TEAL,'s')]:
        m=r['mean_'+key];lo,hi=r['ci95_'+key]
        b.errorbar(m,i+offset,xerr=[[m-lo],[hi-m]],fmt=marker,ms=4,mfc=color,mec=color,
                   ecolor=color,elinewidth=.8,capsize=2,zorder=3)
b.set(ylim=(5.55,-.6),yticks=range(6),yticklabels=regions,xlim=(0,1),xticks=[0,.25,.5,.75,1],xlabel='Mean Dice')
b.xaxis.grid(True,color=GRID,lw=.6);b.set_axisbelow(True)
b.tick_params(axis='y',length=0,pad=5)
b.text(0,1.15,'B',weight='bold',fontsize=12,transform=b.transAxes)
b.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',ms=4,label='All cases'),
                  Line2D([],[],color=TEAL,marker='s',ls='',ms=4,label='Reference-positive')],
         loc='lower left',bbox_to_anchor=(.09,1.095),ncol=2,frameon=False,borderaxespad=0,
         handlelength=.9,handletextpad=.4,columnspacing=.8,labelspacing=.3,fontsize=8.5)
matrix=np.array(d['source_positive_mean']); n=np.array(d['source_positive_n'])
cmap=LinearSegmentedColormap.from_list('regional',['#F3F6F8','#A8C0D1',BLUE])
im=c.imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect='auto')
c.set(xticks=range(6),xticklabels=regions,yticks=range(3),yticklabels=d['source_labels'])
c.tick_params(which='both',length=0,pad=5,labelsize=8.5)
for i in range(3):
    for j in range(6):
        c.text(j,i-.14,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if matrix[i,j]>.66 else INK)
        c.text(j,i+.20,f'n={n[i,j]}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.66 else INK)
c.set_xticks(np.arange(7)-.5,minor=True);c.set_yticks(np.arange(4)-.5,minor=True)
c.grid(which='minor',color='white',lw=1.5)
for s in c.spines.values():s.set_visible(False)
c.text(0,1.15,'C',weight='bold',fontsize=12,transform=c.transAxes)
c.text(.10,1.15,'Source profiles',fontsize=10.5,transform=c.transAxes)
cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(length=0,pad=3,labelsize=7.5);cb.outline.set_visible(False)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
for t in fig.findobj(matplotlib.text.Text):
    if t.get_visible() and t.get_text():
        box=t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1>fig.bbox.width+1 or box.y1>fig.bbox.height+1:outside.append(t.get_text())

for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure2_regional.{ext}',dpi=600,facecolor='white')
plt.close(fig)
