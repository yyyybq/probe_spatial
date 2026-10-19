#!/usr/bin/env python3
"""Rebuild compact paper panels from archived CSVs; no training or inference."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/probe-paper-matplotlib')
from pathlib import Path
import csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/exp_figures/compact_layout'
DATA=Path('results/analysis/q00_1096_v1_mask_validated_20260924_r2/data')
D3=Path('experiments/d_series_scannetpp_v1/diagnostics/dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1/across_seed_metrics.csv')
ACTION=Path('results/independent_goal_conditioned_continuous_action_p8_relational_v2_20260923/continuous_action_paired_effects.csv')
INK='#4E4842'; BLUE='#5C7B86'; BROWN='#926860'; GRAY='#A8A096'; GREEN='#70846F'; GRID='#E8E4DE'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':GRID,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.default':'regular'})
USED={}
def load(path):
 USED[str(path)]={'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
 with (ROOT/path).open() as f:return list(csv.DictReader(f))
def one(rows,**q):
 found=[r for r in rows if all(r[k]==str(v) for k,v in q.items())]
 assert len(found)==1,(q,len(found))
 return found[0]
def save(fig,name):
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
def style(ax):
 for x in ['top','right','left']:ax.spines[x].set_visible(False)
 ax.tick_params(length=0,pad=3,labelsize=7)
 ax.grid(axis='x',color=GRID,linewidth=.5)

def peaks():
 rows=load(DATA/'fig03_layer_statistics.csv')
 models=[('videomae','VideoMAE'),('vjepa2','V-JEPA2'),('wan','Wan'),('cogvideox','CogVideoX')]
 tasks=['A1 Geometry','A1 Mask','A2 View','B2 Persistence','C1 Dynamics','C2 Integration']
 values=[];layers=[];single=[]
 for model,_ in models:
  rr=[one(rows,model=model,task=t,record_type='model_task_peak') for t in tasks]
  single.append([float(r['n_layers']) == 1 for r in rr])
  values.append([float(r['peak_normalized_depth']) for r in rr]);layers.append([int(float(r['peak_layer'])) for r in rr])
 fig=plt.figure(figsize=(2.7,2.55));ax=fig.add_axes([.26,.42,.71,.52])
 cmap=LinearSegmentedColormap.from_list('depth',['#F1F1EE','#CEB8A0',BROWN])
 cmap.set_bad('#F1F1EE')
 im=ax.imshow(np.ma.array(values,mask=single),vmin=0,vmax=1,cmap=cmap,aspect='auto')
 ax.set_yticks(range(4),[n for _,n in models],fontsize=7.3)
 ax.set_xticks(range(6),['Metric geom.','Instance ID','View consist.','Out-of-view loc.','One-step pred.','Path integ.'],rotation=45,ha='right',fontsize=5.8)
 ax.tick_params(length=0,pad=3)
 for y in range(4):
  for x in range(6):ax.text(x,y,str(layers[y][x])+('*' if single[y][x] else ''),ha='center',va='center',fontsize=7.5,color='white' if values[y][x]>.85 and not single[y][x] else INK)
 ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
 ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.34,.10,.5,.035]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1])
 cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=0,pad=2)
 cb.set_label('Color: normalized block depth',fontsize=6.5,labelpad=2)
 fig.text(.61,.98,'Block index (* single measurement)',ha='center',va='top',fontsize=6.7,color=GRAY)
 save(fig,'layer_peaks_half')
 return {'models':[m for m,_ in models],'tasks':tasks,'normalized_depth':values,'layer_indices':layers,'single_measured_layer':single}

def prefixes():
 rows=load(DATA/'fig04_prefix_statistics.csv')
 models=[('dino','DINO'),('f3r','Fast3R'),('videomae','VideoMAE'),('vjepa2','V-JEPA2'),('qwen2_5_vl','Qwen-VL'),('wan','Wan'),('cogvideox','CogVideoX'),('aether','Aether')]
 tasks=['A1 Depth','A1 Mask','A1 Camera','A2 View','B2 Persistence','C1 Dynamics','C2 Integration']
 matrix=[]
 for model,_ in models:
  matrix.append([float(one(rows,model=model,task=t,record_type='model_task_endpoint_status')['standardized_delta'] or 'nan') for t in tasks])
 z=np.array(matrix);assert np.sum(z<0)==45 and np.sum(z>0)==10,(np.sum(z<0),np.sum(z>0))
 bound=np.nanmax(abs(z));norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
 cmap=LinearSegmentedColormap.from_list('prefix',[BROWN,'#F4F3F0',BLUE]);cmap.set_bad('white')
 fig=plt.figure(figsize=(2.7,2.55));ax=fig.add_axes([.265,.42,.705,.52])
 im=ax.imshow(z,cmap=cmap,norm=norm,aspect='auto')
 ax.set_yticks(range(8),[n for _,n in models],fontsize=6.8)
 ax.set_xticks(range(7),['Depth','Instance ID','Camera','View consist.','Out-of-view loc.','One-step pred.','Path integ.'],rotation=45,ha='right',fontsize=5.6)
 ax.tick_params(length=0,pad=2)
 for y in range(8):
  for x in range(7):
   v=z[y,x];mark='NA' if np.isnan(v) else '+' if v>0 else '−' if v<0 else '0'
   ax.text(x,y,mark,fontsize=5.5 if np.isnan(v) else 7,ha='center',va='center',color='white' if abs(v)>bound*.72 else INK)
 ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True)
 ax.grid(which='minor',color='white',linewidth=.65);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.34,.10,.5,.035]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-2,0,2])
 cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=0,pad=2)
 cb.set_label('Standardized change (p24 − p8)',fontsize=6.5,labelpad=2)
 fig.text(.61,.985,'+ longer history helps; − hurts',ha='center',va='top',fontsize=6.7,color=GRAY)
 save(fig,'prefix_effects_half')
 return {'models':[m for m,_ in models],'tasks':tasks,'standardized_delta':[[None if np.isnan(v) else v for v in row] for row in z]}

def prediction():
 rows=load(D3);conds=['native_target_reference','frozen_readout_transfer','retrained_same_capacity_readout','last_context']
 fig=plt.figure(figsize=(2.7,2.5));ax=fig.add_axes([.285,.26,.68,.57]);style(ax)
 ax.set(xlim=(.168,.317),ylim=(-.55,3.55),xticks=[.18,.24,.30])
 ax.set_yticks(range(4),['Copy /\nfrozen','Predicted /\nretrained','Predicted /\nfrozen','Real /\nfrozen'],fontsize=7)
 saved=[]
 for h,color,marker,offset in [(2,BLUE,'o',.12),(4,BROWN,'s',-.12)]:
  prior=one(rows,experiment='D3',reference='native_target',branch='spatial',horizon=h,condition='train_layout_prior')
  ax.axvline(float(prior['mae_mean_across_seeds']),color=color,ls='--',lw=.8,alpha=.8)
  for j,c in enumerate(conds):
   r=one(rows,experiment='D3',reference='native_target',branch='spatial',horizon=h,condition=c)
   v=float(r['mae_mean_across_seeds']);sd=float(r['mae_sd_across_seeds'])
   ax.errorbar(v,3-j+offset,xerr=sd,fmt=marker,ms=3.5,color=color,lw=1,capsize=1.5,zorder=3)
   saved.append({'horizon':h,'condition':c,'mean':v,'sd':sd})
 ax.set_xlabel('Normalized-depth MAE ↓',fontsize=7.2,labelpad=5)
 fig.legend([Line2D([],[],color=BLUE,marker='o',lw=0,ms=4),Line2D([],[],color=BROWN,marker='s',lw=0,ms=4)],['h = 2','h = 4'],loc='upper center',bbox_to_anchor=(.62,.98),ncol=2,frameon=False,fontsize=7,handlelength=.9,columnspacing=1.1)
 fig.text(.6,.095,'Dashed: train-layout prior',ha='center',fontsize=6.7,color=GRAY)
 fig.text(.6,.045,'Error bars: three-seed SD',ha='center',fontsize=6.7,color=GRAY)
 save(fig,'predictive_interface_half');return saved

def action():
 rows=load(ACTION);models=[('dino','DINO'),('f3r','Fast3R'),('vjepa2','V-JEPA2'),('wan','Wan')]
 comparisons=[('full_relational_minus_goal_only','Full − goal only',BLUE,'o',.21),('full_relational_minus_current_only','Full − current only',BROWN,'s',0),('correct_goal_minus_inference_shuffled_goal','Correct − shuffled goal',GREEN,'^',-.21)]
 fig=plt.figure(figsize=(2.7,2.5));ax=fig.add_axes([.245,.22,.725,.52]);style(ax)
 ax.set(xlim=(-9,22),ylim=(-.5,3.5),xticks=[-5,0,10,20])
 ax.set_yticks(range(4),[n for _,n in models][::-1],fontsize=7.2)
 ax.axvline(0,color=GRAY,ls='--',lw=.9)
 saved=[]
 for comparison,label,color,marker,offset in comparisons:
  for j,(m,_) in enumerate(models):
   r=one(rows,model=m,comparison=comparison,metric='positive_progress')
   v,lo,hi=[100*float(r[k]) for k in ['estimate','ci_low','ci_high']]
   assert int(r['n_samples'])==256 and int(r['n_scenes'])==108
   ax.errorbar(v,3-j+offset,xerr=[[v-lo],[hi-v]],fmt=marker,ms=3.1,color=color,lw=.9,capsize=1.3,zorder=3)
   saved.append({'model':m,'comparison':comparison,'metric':'positive_progress','estimate_pp':v,'ci_low_pp':lo,'ci_high_pp':hi})
 handles=[Line2D([],[],color=c,marker=m,lw=.9,ms=3.2) for _,_,c,m,_ in comparisons]
 fig.legend(handles,[x[1] for x in comparisons],loc='upper left',bbox_to_anchor=(.20,1.005),fontsize=6.8,frameon=False,labelspacing=.35,handlelength=1.3)
 ax.set_xlabel('Positive-progress difference (pp)',fontsize=7.1,labelpad=5)
 fig.text(.60,.06,'Relational V2 · paired 95% CI',ha='center',fontsize=6.7,color=GRAY)
 save(fig,'goal_conditioning_half');return saved

def proof():
 fig=plt.figure(figsize=(5.5,5.4))
 for y,title,names in [(.53,'Layer and history diagnostics',['layer_peaks_half','prefix_effects_half']),(.045,'Predictive interface and action controls',['predictive_interface_half','goal_conditioning_half'])]:
  fig.text(.025,y+.43,title,fontsize=10,weight='bold')
  for x,n in zip([.01,.505],names):
   ax=fig.add_axes([x,y,.485,.41]);ax.imshow(plt.imread(OUT/(n+'.png')));ax.axis('off')
 fig.text(.5,.005,'Panel arrangement preview; not a compiled manuscript page',ha='center',fontsize=6,color=GRAY)
 save(fig,'layout_preview')

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 values={'layer_peaks':peaks(),'prefix_effects':prefixes(),'prediction':prediction(),'action':action()}
 proof()
 (OUT/'plot_data.json').write_text(json.dumps({'sources':USED,'data':values},indent=2)+'\n')
 print('Built four half-width panels and layout preview.')
