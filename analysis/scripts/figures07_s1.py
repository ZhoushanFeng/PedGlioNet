import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FIG=OUT=ROOT/'output/figures'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'pdf.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    plan={'paired_dice':json.loads((ROOT/'output/planning_dice.json').read_text())}
    rows=[next(r for r in plan['paired_dice'] if r['scope']=='pooled' and r['region']==region and r['subset']==subset) for region,subset in [('WT','all'),('TC','all'),('ET','positive')]]
    fig,ax=plt.subplots(figsize=(5.0,2.6)); fig.subplots_adjust(left=.16,right=.96,bottom=.25,top=.94)
    for y,r in zip([2,1,0],rows):
        d=r['difference_new_minus_historical'];v=d['value']
        ax.errorbar(v,y,xerr=[[v-d['ci_low']],[d['ci_high']-v]],fmt='o',color='#2F6B91',capsize=4)
    ax.set(yticks=[2,1,0],yticklabels=['WT','TC','ET positive'],xlabel='Development-only minus historical Dice',xlim=(-.032,.012),ylim=(-.5,2.5))
    ax.axvline(0,ls='--',color='gray',lw=.8);ax.grid(axis='x',alpha=.2)
    for ext in ['pdf','svg']:fig.savefig(FIG/f'Figure7_sensitivity.{ext}')
    fig.savefig(OUT/'Figure7_preview.png',dpi=180)
    plt.close(fig)
    p1=json.loads((ROOT/'figure_data/p1_reference.json').read_text())
    methods=[('b0_ensemble','B0','#2F6B91'),('p1_ungated','P1','#C87531'),('p1_gated','P1 with gate','#795A8E')]
    rows=[next(r for r in p1['records'] if r['scope']=='pooled' and r['method']==m) for m,_,_ in methods]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8));fig.subplots_adjust(left=.14,right=.98,bottom=.22,top=.83,wspace=.70)
    for ax,key,label,letter in zip(axes,['positive_dice','negative_specificity'],['ET-positive Dice','ET-negative specificity'],['A','B']):
        for y,row,(_,_,c) in zip([2,1,0],rows,methods):
            d=row[key];v=d['value'];ax.errorbar(v,y,xerr=[[v-d['ci_low']],[d['ci_high']-v]],fmt='o',color=c,capsize=4)
        ax.set(yticks=[2,1,0],yticklabels=[m[1] for m in methods],xlabel=label,xlim=(0,1),ylim=(-.5,2.5));ax.grid(axis='x',alpha=.2)
        ax.text(-.2,1.12,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    for ext in ['pdf','svg']:fig.savefig(FIG/f'FigureS1_p1.{ext}')
    fig.savefig(OUT/'FigureS1_preview.png',dpi=180);plt.close(fig)

if __name__ == '__main__':
    main()
