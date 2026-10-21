"""Run angle x N x probe comparisons and produce per-image reports."""
import argparse, hashlib, json, sys
from pathlib import Path
import experiment as ex


def report(output):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output); df=pd.read_csv(out/'results.csv')
    keys=['object','n','window','sigma','known_probe','rotation','angle']
    m=df.groupby(keys)[['psnr','ssim','complex_nrmse','amplitude_rmse']].agg(['mean','std']).reset_index()
    m.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in m.columns]
    m.to_csv(out/'per_image_metrics.csv',index=False)
    # Paired differences; seed mean is taken within object, objects then equal weighted.
    k=['object','n','window','sigma','known_probe','rotation','seed']
    a=df[np.isclose(df.angle,127.5)].set_index(k)
    b=df[np.isclose(df.angle,ex.GOLDEN)].set_index(k)
    delta=a[['psnr','ssim','complex_nrmse']]-b[['psnr','ssim','complex_nrmse']]
    delta=delta.rename(columns=lambda x:'delta_'+x).reset_index()
    delta.to_csv(out/'paired_differences.csv',index=False)
    obj=delta.groupby(k[:-1]).mean(numeric_only=True).reset_index()
    obj.groupby(['n','window','sigma','known_probe','rotation'])[['delta_psnr','delta_ssim','delta_complex_nrmse']].mean().to_csv(out/'mean_paired_differences.csv')
    images=out/'comparison_images'; images.mkdir(exist_ok=True)
    curves=out/'metric_curves'; curves.mkdir(exist_ok=True)
    # Representative seed is predetermined (smallest seed), not selected for quality.
    seed=int(df.seed.min())
    for condition,group in df[df.seed==seed].groupby(['object','n','window','sigma','known_probe','rotation']):
        group=group.sort_values('angle'); arrays=[]
        for row in group.itertuples():
            with np.load(out/'runs'/row.job_id/'arrays.npz') as z:
                est=z['object'];truth=z['truth']; c=np.vdot(est,truth)/max(np.vdot(est,est).real,1e-30)
                arrays.append((row,truth,c*est))
        fig,axes=plt.subplots(2,3,figsize=(11,7))
        axes[0,0].imshow(abs(arrays[0][1]),cmap='gray',vmin=0,vmax=1);axes[0,0].set_title('True amplitude')
        axes[1,0].imshow(np.angle(arrays[0][1]),cmap='twilight',vmin=-np.pi,vmax=np.pi);axes[1,0].set_title('True phase')
        for col,(row,truth,est) in enumerate(arrays,1):
            axes[0,col].imshow(abs(est),cmap='gray',vmin=0,vmax=1)
            label='127.5 deg' if abs(row.angle-127.5)<1e-8 else 'Golden angle'
            axes[0,col].set_title(f'{label}\nPSNR {row.psnr:.2f} dB | SSIM {row.ssim:.4f}')
            im=axes[1,col].imshow(abs(est-truth),cmap='magma',vmin=0,vmax=1)
            axes[1,col].set_title(f'Complex error | NRMSE {row.complex_nrmse:.4f}')
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'{Path(condition[0]).name} | N={condition[1]} | window={condition[2]} | sigma={condition[3]} | known={bool(condition[4])} | rotation={condition[5]} | seed={seed}');fig.tight_layout()
        tag=hashlib.sha256(str(condition).encode()).hexdigest()[:12]
        fig.savefig(images/f'{tag}.png',dpi=150);plt.close(fig)
    for condition,part in df.groupby(['object','window','sigma','known_probe','rotation']):
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for angle,g in part.groupby('angle'):
            means=g.groupby('n')[['psnr','ssim','complex_nrmse']].mean()
            for ax,metric in zip(axes,['psnr','ssim','complex_nrmse']):
                ax.plot(means.index,means[metric],'o-',label='127.5 deg' if abs(angle-127.5)<1e-8 else 'Golden')
                ax.set(xlabel='Scan positions',ylabel=metric);ax.legend()
        fig.suptitle(' | '.join([Path(str(condition[0])).name]+[str(v) for v in condition[1:]]));fig.tight_layout()
        fig.savefig(curves/(hashlib.sha256(str(condition).encode()).hexdigest()[:12]+'.png'),dpi=150);plt.close(fig)
    # Probe-width response at each N and image.
    for condition,part in df.groupby(['object','n','known_probe','rotation']):
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for angle,g in part.groupby('angle'):
            means=g.groupby('sigma')[['psnr','ssim','complex_nrmse']].mean()
            for ax,metric in zip(axes,['psnr','ssim','complex_nrmse']):
                ax.plot(means.index,means[metric],'o-',label='127.5 deg' if abs(angle-127.5)<1e-8 else 'Golden')
                ax.set(xlabel='Probe sigma (pixels)',ylabel=metric);ax.legend()
        fig.suptitle(' | '.join([Path(str(condition[0])).name]+[str(v) for v in condition[1:]]));fig.tight_layout()
        fig.savefig(curves/('probe_'+hashlib.sha256(str(condition).encode()).hexdigest()[:12]+'.png'),dpi=150);plt.close(fig)
    print(f'Report complete: {out}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='quick.json');p.add_argument('--images',help='Folder of local PNG/JPG/TIF/BMP or .npy amplitude images');p.add_argument('--auto-count',type=int,default=24);p.add_argument('--variants',type=int,default=8);p.add_argument('--workers',type=int,default=2);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    if a.images:
        files=sorted(f.resolve() for f in Path(a.images).iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff','.bmp','.npy'])
        if not files:raise ValueError('No supported images found')
        cfg['objects']=[str(f) for f in files]
        cfg['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    else:
        from dataset import generate
        cfg['objects']=generate(Path(cfg['output'])/'auto_dataset',count=a.auto_count,size=cfg['roi'],variants=a.variants)
        cfg['input_sha256']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in cfg['objects']}
    cfg['source_sha256']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['experiment.py','run.py','dataset.py']}
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    effective=out/'effective_config.json'
    if effective.exists() and json.loads(effective.read_text())!=cfg:raise ValueError('Configuration/source/images changed: choose a new output directory')
    effective.write_text(json.dumps(cfg,indent=2))
    if not a.report_only:
        ex.execute(argparse.Namespace(command='geometry',config=str(effective),workers=1))
        ex.execute(argparse.Namespace(command='reconstruct',config=str(effective),workers=a.workers))
    report(out)

if __name__=='__main__':main()
