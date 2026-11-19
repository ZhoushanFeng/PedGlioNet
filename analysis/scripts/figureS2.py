"""Exploratory, source-coloured PCoA of actual B0 ensemble performance."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output/pcoa'
OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'data/eval'
files=[SRC/f'ensemble_b0_fold{i}_per_case.csv' for i in range(3)]
raw=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
assert len(raw)==257*6 and not raw.duplicated(['case_id','region']).any()
ref=raw[raw.region.eq('ET')].set_index('case_id')
ids=sorted(ref.index[ref.reference_nonempty.eq(True)])
assert len(ids)==171
selected=raw[raw.case_id.isin(ids)&raw.region.isin(['WT','ET'])]
assert selected.reference_nonempty.all() and len(selected)==342
features=pd.DataFrame(index=pd.Index(ids,name='case_id'))
for region in ['WT','ET']:
    for metric in ['dice','nsd_1mm']:
        features[f'{region}_{metric}']=selected[selected.region.eq(region)].set_index('case_id')[metric].reindex(ids)
assert not features.isna().any().any() and ((features>=0)&(features<=1)).all().all()
source=ref.source.reindex(ids).replace({'DFCI-BCH-BWH-PEDs-HGG':'DFCI'})
order=['CBTN','DFCI','DIPGr']; counts=source.value_counts().reindex(order)
assert counts.tolist()==[87,35,49]
means=features.mean(); sds=features.std(ddof=1)
z=((features-means)/sds).to_numpy()
dist=squareform(pdist(z,'euclidean'))
j=np.eye(len(ids))-np.ones((len(ids),len(ids)))/len(ids)
# Explicit contractions avoid version-specific matrix multiply diagnostics on Apple BLAS.
gram = -.5 * np.einsum('ij,jk->ik', np.einsum('ij,jk->ik', j, dist**2, optimize=False), j, optimize=False)
eig,vec=np.linalg.eigh(gram); idx=np.argsort(eig)[::-1];eig=eig[idx];vec=vec[:,idx]
tol=max(1.,eig[0])*1e-10
assert eig.min()>-tol
positive=eig>tol
all_coords=vec[:,positive]*np.sqrt(eig[positive])
assert np.allclose(squareform(pdist(all_coords)),dist,atol=1e-10)
fraction=eig[positive]/eig[positive].sum()
assert np.allclose(fraction[:2],[.615285161722,.324539004552],atol=1e-10)
_,singular,_=np.linalg.svd(z,full_matrices=False)
assert np.allclose(eig[positive],singular**2,atol=1e-10)
xy=all_coords[:,:2].copy()
# Axis signs have no statistical meaning. Orient them deterministically.
for k,anchor in [(0,features.mean(axis=1)),(1,features[['ET_dice','ET_nsd_1mm']].mean(axis=1)-features[['WT_dice','WT_nsd_1mm']].mean(axis=1))]:
    if np.corrcoef(xy[:,k],anchor)[0,1]<0:xy[:,k]*=-1
coords=features.copy();coords.insert(0,'source',source);coords['PCo1']=xy[:,0];coords['PCo2']=xy[:,1]
coords.to_csv(OUT/'case_features_and_coordinates.csv',float_format='%.15g')

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.5,'axes.titlesize':11,
 'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,'axes.unicode_minus':False,
 'text.color':'#26313B','axes.labelcolor':'#26313B','xtick.color':'#26313B','ytick.color':'#26313B'})
colors={'CBTN':'#DF9F4F','DFCI':'#428FAB','DIPGr':'#BA6D65'}
# Preserve the physical plot dimensions while removing title/footer space.
canvas_height=6.0
main_bottom=.6/canvas_height; main_height=4.095/canvas_height
fig=plt.figure(figsize=(7.2,canvas_height))
ax=fig.add_axes([.125,main_bottom,.60,main_height])
top=fig.add_axes([.125,4.835/canvas_height,.60,.98/canvas_height],sharex=ax)
right=fig.add_axes([.747,main_bottom,.16,main_height],sharey=ax)
ellipse_meta={}; extents=[xy.min(axis=0),xy.max(axis=0)]
for group in order:
    points=xy[source.to_numpy()==group]; mean=points.mean(axis=0); cov=np.cov(points,rowvar=False)
    vals,vectors=np.linalg.eigh(cov); ind=np.argsort(vals)[::-1];vals=vals[ind];vectors=vectors[:,ind]
    angle=np.degrees(np.arctan2(vectors[1,0],vectors[0,0])); factor=chi2.ppf(.95,2)
    ax.add_patch(Ellipse(mean,width=2*np.sqrt(factor*vals[0]),height=2*np.sqrt(factor*vals[1]),
                       angle=angle,facecolor='none',edgecolor=colors[group],lw=1.3,zorder=2))
    ax.scatter(points[:,0],points[:,1],s=16,c=colors[group],alpha=.77,lw=.25,edgecolors='white',zorder=3)
    reach=np.sqrt(factor*np.diag(cov));extents.extend([mean-reach,mean+reach])
    ellipse_meta[group]={'n':len(points),'mean':mean.tolist(),'covariance':cov.tolist(),
                         'nominal_normal_reference_probability':.95,'empirical_fraction_inside':float(np.mean(np.einsum('ij,jk,ik->i',points-mean,np.linalg.inv(cov),points-mean)<=factor))}
    pos=order.index(group)
    bp=top.boxplot([points[:,0]],positions=[3-pos],vert=False,widths=.62,
       patch_artist=True,manage_ticks=False,whis=1.5,showfliers=True,
       boxprops={'facecolor':colors[group],'edgecolor':'#44505A','linewidth':.8},
       medianprops={'color':'#29343F','linewidth':1.0},whiskerprops={'color':'#44505A','linewidth':.8},
       capprops={'color':'#44505A','linewidth':.8},flierprops={'marker':'.','markersize':2.5,'markerfacecolor':colors[group],'markeredgecolor':colors[group]})
    right.boxplot([points[:,1]],positions=[pos+1],widths=.62,patch_artist=True,manage_ticks=False,
       whis=1.5,showfliers=True,boxprops={'facecolor':colors[group],'edgecolor':'#44505A','linewidth':.8},
       medianprops={'color':'#29343F','linewidth':1.0},whiskerprops={'color':'#44505A','linewidth':.8},
       capprops={'color':'#44505A','linewidth':.8},flierprops={'marker':'.','markersize':2.5,'markerfacecolor':colors[group],'markeredgecolor':colors[group]})
extents=np.asarray(extents);lo=extents.min(axis=0);hi=extents.max(axis=0);center=(lo+hi)/2;half=(hi-lo)*.545
ratio=(7.2*.60)/4.095; half[0]=max(half[0],half[1]*ratio);half[1]=half[0]/ratio
ax.set(xlim=(center[0]-half[0],center[0]+half[0]),ylim=(center[1]-half[1],center[1]+half[1]),
       xlabel=f'PCo1 ({fraction[0]*100:.2f}%)',ylabel=f'PCo2 ({fraction[1]*100:.2f}%)')
ax.set_aspect('equal',adjustable='box')
ax.axhline(0,c='#D4DADD',lw=.7,zorder=0);ax.axvline(0,c='#D4DADD',lw=.7,zorder=0)
top.set(ylim=(.5,3.5),yticks=[3,2,1],yticklabels=order)
top.tick_params(axis='x',labelbottom=False,length=0);top.tick_params(axis='y',length=0,pad=5,labelsize=9)
top.spines['left'].set_visible(False)
right.set(xlim=(.5,3.5),xticks=[]);right.tick_params(axis='y',labelleft=False,length=0)
ax.legend(handles=[Line2D([],[],ls='',marker='o',color=colors[g],markersize=5,label=f'{g} (n = {counts[g]})') for g in order],
          loc='lower right',frameon=False,fontsize=9,handletextpad=.4)
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'FigureS2_pcoa.{ext}',dpi=600,facecolor='white')
plt.close(fig)
metadata={'analysis':'Exploratory PCoA; not a replacement for paired error-outcome comparisons',
 'n_examinations':171,'group_counts':counts.to_dict(),'model':'Historical B0 three-seed probability ensemble',
 'feature_columns':list(features.columns),'feature_means':means.to_dict(),'feature_sample_sd':sds.to_dict(),
 'standardization':'Pooled case-level mean and sample SD (ddof=1); no source-specific transformation',
 'distance':'Euclidean','ordination':'Classical PCoA of -0.5 J D^2 J; equivalent to PCA of the same standardized features',
 'positive_eigenvalues':eig[positive].tolist(),'minimum_eigenvalue':float(eig.min()),
 'variance_fraction':fraction.tolist(),'first_two_fraction':float(fraction[:2].sum()),
 'distance_reconstruction_max_abs_error':float(np.max(np.abs(squareform(pdist(all_coords))-dist))),
 'ellipse_definition':'(x-mean)^T sample_cov^-1 (x-mean) = chi2(df=2, probability=0.95); descriptive normal-reference contour, not a group-mean CI or a guaranteed 95% empirical coverage region',
 'ellipses':ellipse_meta,'boxplots':'Median, quartiles, whiskers to 1.5 IQR; outlying coordinates shown',
 'limitations':['Source is confounded with held-out fold/model and case mix.','WT includes ET; Dice and boundary-overlap metrics are correlated.','No inference about biological subtypes or image domain separation.','No hypothesis tests or p values performed.'],
 'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
 'method_reference':'https://scikit.bio/docs/latest/generated/skbio.stats.ordination.pcoa.html',
 'validation':{'no_missing':True,'WT_and_ET_reference_nonempty_for_all_cases':True,'matched_independent_PC1_PC2_values':True,'PCA_singular_values_match':True,'manuscript_modified':False}}
(OUT/'analysis_provenance.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
caption=('Exploratory principal coordinate analysis of regional segmentation performance in 171 reference ET-positive held-out MRI examinations. Each point represents one examination evaluated with the historical B0 probability ensemble. Four performance features (WT Dice, WT 1-mm boundary-voxel overlap, ET Dice and ET 1-mm boundary-voxel overlap) were standardized across all included examinations using their sample standard deviations; Euclidean distances were then embedded by classical PCoA. The first two coordinates explain '+f'{100*fraction[:2].sum():.2f}%'+ ' of the total positive eigenvalue sum. WT and ET references are nonempty in every included examination, and no features are missing. Colors identify CBTN (87), DFCI (35) and DIPGr (49). Ellipses are descriptive covariance contours scaled by the 95th percentile of a chi-square distribution with two degrees of freedom; they are not confidence intervals for group means and do not guarantee 95% empirical coverage. Marginal boxplots summarize coordinate medians, quartiles and 1.5-IQR whiskers, with outlying values shown. This Euclidean PCoA is equivalent to PCA of the standardized feature matrix. Correlated metrics and nested regions mean the four features are not independent dimensions. Source is tied to the held-out fold/model; visual grouping cannot establish a pure source effect, image-domain separation or biological subtypes. No group significance tests were performed. This exploratory display addresses multivariate performance patterns and does not replace the paired B0-versus-AdultInit error analyses in Figure 4.')
(OUT/'caption.txt').write_text(caption+'\n')
print(json.dumps({'n':171,'counts':counts.to_dict(),'first_two_percent':(fraction[:2]*100).tolist(),'total_percent':float(fraction[:2].sum()*100)},indent=2))
