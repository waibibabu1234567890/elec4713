"""Independent quasi-Fermat geometry and ePIE study. See README_zh.md."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse, csv, hashlib, itertools, json, math, platform, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import zoom, gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

GOLDEN = 180*(3-math.sqrt(5))

def positions(n, radius, angle, rotation=0, design='normalized', reference_n=300):
    denominator = n-1 if design == 'normalized' else reference_n-1
    if n < 2 or denominator < n-1: raise ValueError('N >= 2 and reference_n >= N required')
    r = radius*np.sqrt(np.arange(n)/denominator)
    theta = np.deg2rad(np.arange(n)*angle+rotation)
    return np.column_stack((r*np.sin(theta),r*np.cos(theta))) # y,x

def probe(window, sigma):
    y,x=np.indices((window,window))-(window-1)/2
    p=np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return (p/np.linalg.norm(p)).astype(complex)

def layout(cfg,n,angle,rotation,window):
    roi=cfg['roi']; radius=cfg['radius']
    side=2*int(math.ceil(radius+window+roi/2))+2
    centre=side//2
    xy=positions(n,radius,angle,rotation,cfg.get('design','normalized'),cfg.get('reference_n',300))
    starts=np.rint(xy+centre-window//2).astype(int)
    sl=slice(centre-roi//2,centre-roi//2+roi)
    return side,starts,sl,xy

def geometry(cfg,n,angle,rotation,window,sigma):
    side,starts,sl,xy=layout(cfg,n,angle,rotation,window)
    illumination=np.zeros((side,side)); p=abs(probe(window,sigma))**2
    for y,x in starts: illumination[y:y+window,x:x+window]+=p
    a=illumination[sl,sl]; mean=a.mean()
    y,x=np.indices(a.shape); centre=side//2
    pixels=np.column_stack((y.ravel()+sl.start-centre,x.ravel()+sl.start-centre))
    rounded_centres=starts+window//2-centre
    d=cKDTree(rounded_centres).query(pixels)[0]
    pair=np.linalg.norm(rounded_centres[:,None]-rounded_centres[None,:],axis=2)
    weights=np.exp(-pair**2/(2*sigma*sigma)); np.fill_diagonal(weights,0)
    return dict(coverage=float(np.mean(a>.01*mean)),zero_fraction=float(np.mean(a==0)),
                illumination_cv=float(a.std()/max(mean,1e-15)),max_hole=float(d.max()),
                unique_positions=len(np.unique(starts,axis=0)),weighted_degree=float(weights.sum(axis=1).mean()))

def sample(name,size,mode):
    """Procedural objects; these are not the thesis's original sample images."""
    y,x=np.mgrid[0:1:complex(size),0:1:complex(size)]
    seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    if name=='smooth': a=.1+.9*(.5+.25*np.sin(9*x+2*y)+.25*np.cos(12*y-x))
    elif name=='stripes': a=.1+.9*(np.sin(40*x+5*np.sin(8*y))>0)
    elif name=='sparse':
        a=np.full((size,size),.1)
        for cy,cx in rng.uniform(.1,.9,(20,2)):
            a+=.9*np.exp(-((x-cx)**2+(y-cy)**2)/(.015**2))
        a=np.clip(a,.1,1)
    elif name=='texture':
        a=gaussian_filter(rng.normal(size=(size,size)),1.2); a=.1+.9*(a-a.min())/(np.ptp(a)+1e-15)
    else:
        path=Path(name)
        if path.suffix=='.npy': a=np.load(path)
        else:
            from PIL import Image
            a=np.asarray(Image.open(path).convert('L'),float)/255
        if a.ndim!=2: raise ValueError('Input must be a 2D amplitude image')
        a=zoom(a,(size/a.shape[0],size/a.shape[1]),order=1)[:size,:size]
        if not np.isfinite(a).all() or a.min()<0 or a.max()>1: raise ValueError('Input amplitude must be finite in [0,1]')
    phase=1.5*(a-.1)/.9 # image-derived phase, distinct for each input
    if mode=='amplitude': return a.astype(complex)
    if mode=='phase': return np.exp(1j*phase)
    if mode=='mixed': return a*np.exp(1j*phase)
    raise ValueError(mode)

def metrics(est,true):
    c=np.vdot(est,true)/max(float(np.vdot(est,est).real),1e-30)
    aligned=c*est
    err=np.linalg.norm(aligned-true)/np.linalg.norm(true)
    rmse=np.sqrt(np.mean((abs(aligned)-abs(true))**2))
    # SSIM: uniform 7x7, sample covariance, valid interior, data_range=1.
    a=abs(aligned); b=abs(true); win=min(7,min(a.shape)); win-=1-win%2; factor=win*win/(win*win-1)
    ma=uniform_filter(a,win); mb=uniform_filter(b,win)
    va=factor*(uniform_filter(a*a,win)-ma*ma)
    vb=factor*(uniform_filter(b*b,win)-mb*mb)
    cov=factor*(uniform_filter(a*b,win)-ma*mb)
    ss=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    pad=win//2
    ssim=float(ss[pad:-pad,pad:-pad].mean())
    return dict(complex_nrmse=float(err),amplitude_rmse=float(rmse),psnr=float(-20*np.log10(max(rmse,1e-10))),ssim=ssim)

def reconstruct(cfg,task):
    n,angle,rotation,window,sigma,name,seed,known=task
    key=json.dumps(task,separators=(',',':')); jobid=hashlib.sha256(key.encode()).hexdigest()[:16]
    out=Path(cfg['output'])/'runs'/jobid; out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists(): return json.loads((out/'done.json').read_text())
    side,starts,sl,_=layout(cfg,n,angle,rotation,window)
    truth=np.ones((side,side),complex); truth[sl,sl]=sample(name,cfg['roi'],cfg['mode'])
    ptrue=probe(window,sigma)
    f=lambda a:np.fft.fft2(a,norm='ortho')
    inv=lambda a:np.fft.ifft2(a,norm='ortho')
    noise=np.random.default_rng(np.random.SeedSequence([seed,1])); order=np.random.default_rng(np.random.SeedSequence([seed,2]))
    amps=[]; counts=cfg.get('counts_per_exposure',0)
    if cfg.get('total_detected_counts',0)>0: counts=cfg['total_detected_counts']/n
    for y,x in starts:
        intensity=abs(f(truth[y:y+window,x:x+window]*ptrue))**2
        if counts>0:
            total=intensity.sum(); intensity=noise.poisson(counts*intensity/max(total,1e-30))*total/counts
        amps.append(np.sqrt(intensity))
    obj=np.ones_like(truth); p=ptrue.copy() if known else probe(window,sigma*cfg.get('initial_probe_factor',1.1))
    exterior=np.ones(truth.shape,bool); exterior[sl,sl]=False
    history=[]; elapsed=0.; updates=0
    def record(it):
        history.append(dict(iteration=it,position_updates=updates,update_seconds=elapsed,**metrics(obj[sl,sl],truth[sl,sl])))
    record(0)
    iterations=cfg['iterations']
    if cfg.get('update_budget') is not None: iterations=cfg['update_budget']//n
    for it in range(1,iterations+1):
        permutation=order.permutation(n); tick=time.perf_counter()
        for j in permutation:
            y,x=starts[j]; old=obj[y:y+window,x:x+window].copy()
            exitwave=old*p; z=f(exitwave)
            corrected=inv(amps[j]*np.exp(1j*np.angle(z))); delta=corrected-exitwave
            obj[y:y+window,x:x+window]+=cfg['eta_object']*np.conj(p)*delta/max(float(np.max(abs(p)**2)),1e-15)
            if not known: p+=cfg['eta_probe']*np.conj(old)*delta/max(float(np.max(abs(old)**2)),1e-15)
            obj[exterior]=1 # known unit exterior prior, also between position updates
        elapsed+=time.perf_counter()-tick; updates+=n
        if it%cfg['save_every']==0 or it==iterations: record(it)
    write_csv(out/'history.csv',history)
    if cfg.get('save_arrays',True): np.savez_compressed(out/'arrays.npz',object=obj[sl,sl],truth=truth[sl,sl],probe=p,true_probe=ptrue,starts=starts)
    row=dict(job_id=jobid,n=n,angle=angle,rotation=rotation,window=window,sigma=sigma,object=name,seed=seed,known_probe=known,**history[-1])
    for target in cfg['targets']:
        hit=next((r for r in history if r['complex_nrmse']<=target),None)
        label=f'{target:g}'
        row['hit_iteration_'+label]=hit['iteration'] if hit else ''
        row['hit_seconds_'+label]=hit['update_seconds'] if hit else ''
        row['final_success_'+label]=history[-1]['complex_nrmse']<=target
    temp=out/'done.tmp'; temp.write_text(json.dumps(row,indent=2)); temp.replace(out/'done.json')
    return row

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tasks(cfg):
    for n,a,r,p,name,seed,known in itertools.product(cfg['positions'],cfg['angles'],cfg['rotations'],cfg['probes'],cfg['objects'],cfg['seeds'],cfg['known_probe']):
        yield (n,a,r,p[0],p[1],name,seed,known)

def execute(args):
    cfg=json.loads(Path(args.config).read_text()); out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    frozen=json.dumps(cfg,sort_keys=True,indent=2)
    manifest=out/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=cfg: raise ValueError('Output contains a different configuration. Choose a fresh output path.')
    manifest.write_text(frozen)
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if args.command=='geometry':
        rows=[]
        combinations=list(itertools.product(cfg['positions'],cfg['angles'],cfg['rotations'],cfg['probes']))
        for k,(n,a,r,p) in enumerate(combinations,1):
            rows.append(dict(n=n,angle=a,rotation=r,window=p[0],sigma=p[1],**geometry(cfg,n,a,r,*p)))
            if k%100==0:print(f'Geometry {k}/{len(combinations)}',flush=True)
        write_csv(out/'geometry.csv',rows);print(f'Saved {len(rows)} geometries to {out}')
    else:
        all_tasks=list(tasks(cfg));print(f'{len(all_tasks)} reconstruction records; completed jobs resume automatically',flush=True)
        from functools import partial
        if args.workers==1: iterator=map(partial(reconstruct,cfg),all_tasks); pool=None
        else:
            pool=ProcessPoolExecutor(max_workers=args.workers); iterator=pool.map(partial(reconstruct,cfg),all_tasks)
        rows=[]
        try:
            for row in iterator:
                rows.append(row);print(f'{len(rows)}/{len(all_tasks)} N={row["n"]} angle={row["angle"]:.5f} error={row["complex_nrmse"]:.5g}',flush=True)
        finally:
            if pool:pool.shutdown()
        write_csv(out/'results.csv',rows)
        summarize(out)

def summarize(out):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);df=pd.read_csv(out/'results.csv')
    group=['n','angle','rotation','window','sigma','known_probe']
    # Equal weight per object, after averaging repeated seeds.
    obj=df.groupby(group+['object'],dropna=False)[['complex_nrmse','amplitude_rmse','psnr','ssim']].mean().reset_index()
    obj.to_csv(out/'object_means.csv',index=False)
    summary=obj.groupby(group).agg(mean_nrmse=('complex_nrmse','mean'),object_std=('complex_nrmse','std'),objects=('object','size')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    for target in [c for c in df if c.startswith('hit_iteration_')]:
        target_value=target.removeprefix('hit_iteration_'); g=df.groupby(group)
        stats=g[target].agg(['count','size','median']).rename(columns={'count':'hits','size':'runs','median':'success_conditional_median_iteration'})
        stats['final_successes']=g['final_success_'+target_value].sum()
        stats.to_csv(out/f'attainment_{target_value}.csv')
    for key,part in summary.groupby(['rotation','window','sigma','known_probe']):
        fig,ax=plt.subplots(figsize=(9,5))
        for n,p in part.groupby('n'):
            p=p.sort_values('angle');ax.plot(p.angle,p.mean_nrmse,'.-',label=f'N={n}')
        ax.axvline(GOLDEN,ls=':',color='gray',label='Golden angle')
        ax.set(xlabel='Angular increment (degrees)',ylabel='Mean aligned complex NRMSE',title=f'rotation/window/sigma/known={key}')
        ax.legend();fig.tight_layout();fig.savefig(out/('accuracy_'+hashlib.sha256(str(key).encode()).hexdigest()[:8]+'.png'),dpi=160);plt.close(fig)
    print(f'Summary saved: {out}')

def selftest():
    import tempfile
    p=positions(100,30,127.5);assert np.allclose(p[-1]@p[-1],900)
    p1=positions(100,30,127.5,design='nested',reference_n=300)
    p2=positions(300,30,127.5,design='nested',reference_n=300);assert np.array_equal(p1,p2[:100])
    x=np.arange(1,17).reshape(4,4).astype(complex);assert metrics(x*(2+3j),x)['complex_nrmse']<1e-14
    assert abs(np.sum(abs(probe(16,3))**2)-1)<1e-14
    with tempfile.TemporaryDirectory() as d:
        cfg=dict(output=d,roi=24,radius=18,mode='amplitude',iterations=70,save_every=5,eta_object=.8,eta_probe=.1,targets=[.1],save_arrays=True)
        row=reconstruct(cfg,(80,GOLDEN,0,16,3,'smooth',2026,True))
        h=np.genfromtxt(Path(d)/'runs'/row['job_id']/'history.csv',delimiter=',',names=True,encoding='utf-8-sig')
        assert row['complex_nrmse']<float(h['complex_nrmse'][0])*.2,(row,h[0])
        assert row['complex_nrmse']<.1,row
        row2=reconstruct(cfg,(80,GOLDEN,0,16,3,'smooth',2026,False))
        assert np.isfinite(row2['complex_nrmse'])
        assert reconstruct(cfg,(80,GOLDEN,0,16,3,'smooth',2026,True))==row
    print('PASS: radial law, nested subsets, scalar alignment, probe normalization, known-probe recovery, blind finite output and resume')

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for command in ('geometry','reconstruct'):
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=1)
    p=sub.add_parser('summarize');p.add_argument('--output',required=True)
    sub.add_parser('selftest');args=parser.parse_args()
    if args.command=='selftest':selftest()
    elif args.command=='summarize':summarize(args.output)
    else:execute(args)
