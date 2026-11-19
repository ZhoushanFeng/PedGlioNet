#!/usr/bin/env python3
"""Audited ET omission figures from retrieved observations; no new statistics."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures'
MEMBERSHIP = ROOT / 'data/omission_membership.json'
QUALITY = ROOT / 'figure_data/reliability_discrimination.csv'
SEEDS = ['2026', '2027', '2028']
BLUE, ORANGE, TEAL, RED = '#285A80', '#BC783B', '#32857B', '#B24555'
PALE_BLUE, GRID, INK = '#ECEFF1', '#E6E9EC', '#28343F'
SOURCES = {'CBTN': 'CBTN', 'DFCI-BCH-BWH-PEDs-HGG': 'DFCI', 'DIPGr': 'DIPGr'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'font.size': 9,
        'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'pdf.use14corefonts': False, 'savefig.facecolor': 'white'})


def save(fig, name, data, caption):
    fig.canvas.draw()
    text_sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(text_sizes) >= 7.5, (name, min(text_sizes))
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / (name + '.' + ext), dpi=600, bbox_inches='tight', pad_inches=.10)
    (OUT / (name + '_data.json')).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (OUT / (name + '_caption.txt')).write_text(caption + '\n')
    plt.close(fig)


def paired_matrix(rows, left, right):
    cells = np.zeros((2, 2), dtype=int)
    for r in rows:
        x = 0 if r['all_three_seed_et_empty'][left] else 1
        y = 0 if r['all_three_seed_et_empty'][right] else 1
        cells[x, y] += 1
    return cells


def draw_matrix(ax, cells, xlabel, ylabel, n):
    colors = [[RED, '#F5E6D8'], ['#DFEAF1', '#F0F3F5']]
    for row in range(2):
        for col in range(2):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=colors[row][col], edgecolor='white', lw=2.2))
            ax.text(col, row, str(cells[row, col]), ha='center', va='center', fontsize=18,
                    color='white' if (row, col) == (0, 0) else INK)
    ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), xticks=[0, 1], yticks=[0, 1],
        xticklabels=['All empty', 'Not all empty'], yticklabels=['All empty', 'Not all empty'],
        xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect('equal')
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1, 1.055, f'n = {n}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5)


def figure6(cases):
    pos = [r for r in cases if r['reference_et_positive']]
    changed = [r for r in pos if r['plan_changed']]
    omitted = [r for r in pos if r['all_three_seed_et_empty']['b0']]
    others = [r for r in pos if not r['all_three_seed_et_empty']['b0']]
    assert len(pos) == 171 and len(omitted) == 21
    assert sum(r['ensemble_wt_dice']['historical'] >= .9 for r in omitted) == 11
    assert all(r['ensemble_et_dice']['historical'] == 0 for r in omitted)
    mp = paired_matrix(changed, 'b0', 'development_plan')
    mt = paired_matrix(pos, 'b0', 'adult_init')
    assert mp.tolist() == [[14, 4], [2, 116]]
    assert mt.tolist() == [[15, 6], [2, 148]]
    with QUALITY.open(newline='') as f:
        quality = list(csv.DictReader(f))
    signals = [('mean_entropy', 'Entropy'), ('et_disagreement', 'ET disagreement'),
               ('low_predicted_et_volume', 'Low ET volume')]
    rows = []
    for target, n in [('ET_positive', 171), ('ET_all', 257)]:
        for signal, label in signals:
            selected = [r for r in quality if r['target'] == target and r['scope'] == 'pooled'
                        and r['signal'] == signal and float(r['failure_threshold']) == .5]
            assert len(selected) == 1
            r = selected[0]
            assert int(r['n']) == n
            rows.append({'target': target, 'signal': signal, 'label': label, 'n': n,
                         'failures': int(r['failures']), 'auroc': float(r['auroc_value']),
                         'ci95': [float(r['auroc_ci_low']), float(r['auroc_ci_high'])]})
    fig = plt.figure(figsize=(7.2, 6.15))
    a = fig.add_axes([.09, .605, .345, .325])
    b = fig.add_axes([.665, .605, .315, .325])
    c = fig.add_axes([.19, .12, .235, .285])
    d = fig.add_axes([.69, .12, .235, .285])
    a.axvspan(.9, 1.015, color=BLUE, alpha=.045, zorder=0)
    a.axvline(.9, color='#A9B2BA', lw=.8, ls=(0, (3, 3)), zorder=1)
    a.scatter([r['ensemble_wt_dice']['historical'] for r in others],
              [r['ensemble_et_dice']['historical'] for r in others],
              s=17, marker='o', facecolor=BLUE, edgecolor=BLUE, linewidth=.4, alpha=.48, zorder=3)
    a.scatter([r['ensemble_wt_dice']['historical'] for r in omitted],
              [r['ensemble_et_dice']['historical'] for r in omitted],
              s=36, marker='^', facecolor=RED, edgecolor='white', linewidth=.4, alpha=.88, zorder=4)
    a.set(xlim=(-.025, 1.025), ylim=(-.035, 1.025), xticks=np.linspace(0, 1, 6),
          yticks=np.linspace(0, 1, 6), xlabel='Ensemble WT Dice', ylabel='Ensemble ET Dice')
    a.yaxis.grid(color=GRID, lw=.55); a.set_axisbelow(True)
    a.text(.03, .95, '171 ET-positive cases', transform=a.transAxes, fontsize=8.5, va='top')
    a.legend(handles=[Line2D([], [], ls='', marker='o', color=BLUE, alpha=.65, ms=4.2, label='At least one seed nonempty'),
                      Line2D([], [], ls='', marker='^', color=RED, ms=5, label='All 3 seeds empty (n = 21)')],
             loc='upper left', bbox_to_anchor=(-.01, -.27), frameon=False, borderpad=0,
             handletextpad=.5, labelspacing=.4)
    for target, offset, color, marker in [('ET_positive', -.15, TEAL, 'o'), ('ET_all', .15, BLUE, 's')]:
        for i, (signal, _) in enumerate(signals):
            r = next(r for r in rows if r['target'] == target and r['signal'] == signal)
            mean = r['auroc']; lo, hi = r['ci95']
            b.errorbar(mean, i+offset, xerr=[[mean-lo], [hi-mean]], fmt=marker, ms=4.5,
                       color=color, lw=1.1, capsize=2.2, zorder=3)
    b.axvline(.5, ls=(0, (3, 3)), lw=.8, color='#88939D', zorder=1)
    b.set(xlim=(0, 1.025), ylim=(2.6, -.6), xticks=[0, .25, .5, .75, 1],
          yticks=[0, 1, 2], yticklabels=[label for _, label in signals], xlabel='Failure-detection AUROC')
    b.tick_params(axis='y', length=0, pad=7)
    b.xaxis.grid(color=GRID, lw=.55); b.set_axisbelow(True)
    b.legend(handles=[Line2D([], [], color=TEAL, marker='o', ms=4.2, lw=1, label='ET-positive (n = 171)'),
                      Line2D([], [], color=BLUE, marker='s', ms=4.2, lw=1, label='All cases (n = 257)')],
             loc='upper left', bbox_to_anchor=(-.40, -.27), frameon=False, borderpad=0,
             handletextpad=.6, labelspacing=.4)
    draw_matrix(c, mp, 'Development-only planning', 'Historical planning', 136)
    draw_matrix(d, mt, 'AdultInit', 'B0', 171)
    for x, y, label in [(.03,.96,'A'),(.535,.96,'B'),(.03,.445,'C'),(.535,.445,'D')]:
        fig.text(x, y, label, fontsize=12, fontweight='bold', va='top')
    data = {'figure': 'Figure8_omissions', 'input_sha256': {str(p.relative_to(ROOT)):sha(p) for p in [MEMBERSHIP, QUALITY]},
        'A': {'n':171,'omissions':21,'omissions_wt_ge_0p9':11,'coordinates':'Observed ensemble values; no jitter or synthetic points.',
              'cases':[{'case_id':r['case_id'],'fold':r['fold'],'source':r['source'],'wt_dice':r['ensemble_wt_dice']['historical'],
                        'et_dice':r['ensemble_et_dice']['historical'],'all_three_empty':r['all_three_seed_et_empty']['b0']} for r in pos]},
        'B': {'rows':rows,'intervals':'Original 95% source-stratified case-bootstrap intervals; not recalculated.'},
        'C': {'n':136,'matrix':mp.tolist(),'rows':['historical_all_empty','historical_not_all_empty'],
              'columns':['development_all_empty','development_not_all_empty'],'persistent_high_wt_both':9},
        'D': {'n':171,'matrix':mt.tolist(),'rows':['b0_all_empty','b0_not_all_empty'],
              'columns':['adult_all_empty','adult_not_all_empty']}}
    caption = (
        'Figure 8. Case-level ET omissions and population-dependent quality assessment. '
        '(A) Historical B0 probability-ensemble WT and ET Dice for 171 reference ET-positive examinations. '
        'Circles denote at least one seed with nonempty ET; triangles denote 21 examinations with empty ET in all three seeds. '
        'Coordinates are observed values without jitter; coincident points may overlap. The dashed line and pale band mark WT Dice ≥0.90, '
        'satisfied by 11 of the 21 shared omissions. (B) AUROC for detecting ET Dice <0.5 using entropy, ET pairwise disagreement and low predicted ET volume. '
        'Circles represent the 171 ET-positive examinations (49 failures); squares represent all 257 examinations (60 failures). '
        'Horizontal lines are the original 95% source-stratified case-bootstrap intervals from 2,000 replicates, and the dashed line marks AUROC=0.5. '
        'The reference-positive population requires reference labels. (C,D) Paired counts of all-three-seed empty ET predictions. '
        '“All empty” means every seed predicts empty ET; “Not all empty” means at least one seed predicts nonempty ET. '
        'Cell colors distinguish paired outcomes and do not encode count magnitude. (C) Historical versus development-only planning in 136 ET-positive CBTN and DIPGr examinations: '
        '14 persistent, 4 historical-only, 2 development-only and 116 with no shared omission under either plan. Nine persistent cases have WT Dice ≥0.90 under both probability ensembles. '
        'DFCI is excluded because its predictions were reused. (D) B0 versus AdultInit in all 171 ET-positive examinations: 15 persistent, 6 B0-only, 2 AdultInit-only and 148 with no shared omission under either initialization. '
        'A nonempty prediction records regional presence, not necessarily correct delineation. The two matrices use different comparison populations.')
    save(fig, 'Figure8_omissions', data, caption)


def figure_s3(cases):
    pos = [r for r in cases if r['reference_et_positive']]
    changed = [r for r in pos if r['plan_changed']]
    arows = [r for r in pos if r['all_three_seed_et_empty']['b0'] or r['all_three_seed_et_empty']['adult_init']]
    brows = [r for r in changed if r['all_three_seed_et_empty']['b0'] or r['all_three_seed_et_empty']['development_plan']]
    assert (len(arows), len(brows)) == (23,20)
    assert (len(pos)-len(arows), len(changed)-len(brows)) == (148,116)
    cmap = ListedColormap([PALE_BLUE, RED]); norm = BoundaryNorm([-.5,.5,1.5], cmap.N)
    fig = plt.figure(figsize=(7.2,6.65))
    configurations = [
        ('A',arows,'adult_init','B0','AdultInit',171,[.16,.18,.265,.64],'Adult initialization','Historical WT'),
        ('B',brows,'development_plan','Historical','Development-only',136,[.665,.18,.265,.64],'Planning configuration','Both-plan WT')]
    data = {'figure':'Figure9_omission_heatmap','input_sha256':{str(MEMBERSHIP.relative_to(ROOT)):sha(MEMBERSHIP)},
            'encoding':{'0':'nonempty ET prediction','1':'empty ET prediction in a reference-positive examination'},
            'selection':'Reference ET-positive and all-three-seed empty ET in at least one of the two contrasted conditions.',
            'ordering':'Held-out fold, then full case ID; not ordered by Dice or outcome.', 'panels':{}}
    for letter, selected, right, left_label, right_label, denominator, rect, title, wt_label in configurations:
        ax = fig.add_axes(rect)
        values = np.array([[int(not r['seed_et_presence'][condition][s]) for condition in ['b0',right] for s in SEEDS] for r in selected])
        assert values.shape == (len(selected),6)
        assert all(isinstance(r['seed_et_presence'][condition][seed], bool) for r in selected for condition in ['b0', right] for seed in SEEDS)
        ax.imshow(values,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest',extent=(-.5,5.5,len(selected)-.5,-.5))
        ax.set(xlim=(-.5,7.1),ylim=(len(selected)-.5,-.5),yticks=np.arange(len(selected)),
               yticklabels=[SOURCES[r['source']]+' '+r['case_id'].split('-')[2] for r in selected],
               xticks=np.arange(6),xticklabels=SEEDS+SEEDS)
        ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=3,labelrotation=90,labelsize=8)
        ax.tick_params(axis='y',length=0,pad=5,labelsize=8)
        # Within-condition cell boundaries and an explicit condition divider.
        for x in np.arange(-.5,6,1): ax.plot([x,x],[-.5,len(selected)-.5],color='white',lw=.55,zorder=2)
        for y in np.arange(-.5,len(selected),1): ax.plot([-.5,5.5],[y,y],color='white',lw=.55,zorder=2)
        ax.axvline(2.5,color='white',lw=3,zorder=3)
        # A cross is redundant with red, allowing binary state identification in grayscale.
        yy,xx=np.where(values==1)
        ax.scatter(xx,yy,marker='x',s=9,linewidths=.55,color='white',zorder=4)
        wt_flags=[]
        for i,r in enumerate(selected):
            high=r['ensemble_wt_dice']['historical']>=.9
            if letter=='B': high=high and r['ensemble_wt_dice']['development_plan']>=.9
            wt_flags.append(high)
            ax.plot(6.45,i,marker='D',ms=4.1,color=TEAL if high else '#BAC2C8',
                    markerfacecolor=TEAL if high else 'white',markeredgewidth=.65,ls='',clip_on=False)
            if i>0 and selected[i-1]['source']!=r['source']:
                ax.plot([-.5,6.8],[i-.5,i-.5],color='white',lw=2,zorder=5)
        fig.text(rect[0]+1.5/7.6*rect[2], .895, left_label, ha='center', va='center', fontsize=9)
        fig.text(rect[0]+4.5/7.6*rect[2], .895, right_label if letter=='A' else 'Development\nonly', ha='center', va='center', fontsize=9, linespacing=.95)
        ax.text(6.45,-1.0,'WT ≥0.90',rotation=90,ha='center',va='bottom',fontsize=8,clip_on=False)
        for spine in ax.spines.values():spine.set_visible(False)
        panel_x=.025 if letter=='A' else .53
        fig.text(panel_x,.985,letter,fontsize=12,fontweight='bold',va='top')
        fig.text(panel_x+.04,.985,title,fontsize=10.5,va='top')
        fig.text(panel_x+.04,.947,f'{len(selected)} of {denominator} ET-positive cases',fontsize=9,va='top')
        fig.text(rect[0],.145,wt_label+' ≥0.90: filled diamond',fontsize=8,va='top')
        panel_cases=[]
        for r,states,high in zip(selected,values.tolist(),wt_flags):
            panel_cases.append({'case_id':r['case_id'],'short_id':SOURCES[r['source']]+' '+r['case_id'].split('-')[2],
                'fold':r['fold'],'source':r['source'],'reference_et_volume_mm3':r['reference_et_volume_mm3'],
                'displayed_empty_state':states,'seed_et_presence':{k:r['seed_et_presence'][k] for k in ['b0',right]},
                'all_three_seed_et_empty':{k:r['all_three_seed_et_empty'][k] for k in ['b0',right]},
                'ensemble_wt_dice':r['ensemble_wt_dice'],'high_wt_annotation':bool(high)})
        data['panels'][letter]={'n_displayed':len(selected),'n_reference_positive_population':denominator,
            'n_no_all_seed_empty_event_in_either_condition':denominator-len(selected),'missing_seed_states':0,
            'column_conditions':['b0']*3+[right]*3,'column_seeds':SEEDS+SEEDS,
            'high_wt_rule':'Historical B0 probability-ensemble WT Dice ≥0.90' if letter=='A' else 'Historical and development-only B0 ensemble WT Dice both ≥0.90',
            'cases':panel_cases}
    fig.legend(handles=[Patch(facecolor=RED,edgecolor='white',label='Empty ET (×)'),
                        Patch(facecolor=PALE_BLUE,edgecolor='white',label='Nonempty ET'),
                        Line2D([],[],ls='',marker='D',color=TEAL,ms=4.2,label='WT criterion met')],
               loc='lower center',bbox_to_anchor=(.53,.065),ncol=3,frameon=False,columnspacing=1.9,handlelength=1.1)
    fig.text(.53,.035,'All displayed cases are reference ET-positive. Nonempty ET does not imply accurate segmentation.',ha='center',fontsize=8)
    caption = (
        'Supplementary Figure 9. Per-seed ET presence in examinations with shared omissions. '
        'Each row is one reference ET-positive examination, and each cell is one observed seed prediction. '
        'Red cells with a white cross denote empty ET predictions; light gray cells denote nonempty ET predictions. All 258 displayed seed states are observed, with no missing predictions. '
        'The six columns compare seeds 2026, 2027 and 2028 within each of two conditions; the center gap separates conditions. '
        '(A) B0 versus AdultInit: 23 of 171 ET-positive examinations had empty ET in all three seeds under at least one initialization. '
        'The remaining 148 examinations had no all-three-seed empty state under either initialization and are not displayed. '
        'Fifteen displayed examinations were omitted by all seeds in both conditions, six only in B0 and two only in AdultInit. '
        'A filled diamond indicates historical B0 probability-ensemble WT Dice ≥0.90; AdultInit probability-ensemble WT Dice is not implied. '
        '(B) Historical versus development-only planning in CBTN and DIPGr: 20 of 136 ET-positive examinations had an all-three-seed empty state under at least one plan. '
        'The remaining 116 examinations are not displayed; DFCI is excluded because its predictions were reused. '
        'Fourteen displayed examinations were omitted by all seeds under both plans, four only under historical planning and two only under development-only planning. '
        'A filled diamond indicates WT Dice ≥0.90 under both B0 probability ensembles. Nine of the 14 persistent planning omissions satisfy this criterion. '
        'An unfilled diamond indicates that the specified WT criterion is not met. '
        'Rows are ordered by held-out source and case identifier, without sorting on model performance. Row labels combine source and the five-digit dataset case number; '
        'the complete BraTS-PED identifiers and all plotted values are retained in the accompanying data JSON. '
        'These selected subsets describe shared omissions and cannot estimate overall prevalence without their full denominators. '
        'A transition to a nonempty prediction does not establish correct lesion detection or accurate delineation.')
    save(fig,'Figure9_omission_heatmap',data,caption)


def main():
    setup()
    dataset=json.loads(MEMBERSHIP.read_text())
    cases=dataset['cases']
    assert len(cases)==257 and len({r['case_id'] for r in cases})==257
    cases=sorted(cases,key=lambda r:(r['fold'],r['case_id']))
    figure6(cases)
    figure_s3(cases)
    print('Created Figure8_omissions and Figure9_omission_heatmap; all observed values verified.')


if __name__=='__main__':
    main()
