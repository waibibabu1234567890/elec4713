"""Deterministic morphology-diverse synthetic benchmark; no network required."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,label,convolve

FAMILIES=['solid','islands','rings','perforated','nested_rings','tree','network','labyrinth','fractures','stripes','checkerboard','radial','spiral','blobs','multiscale','gradient','edges','grains','glyphs','scene']

def make(family,seed,size):
    rng=np.random.default_rng(seed);y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    theta=rng.uniform(0,np.pi);u=x*np.cos(theta)+y*np.sin(theta);v=-x*np.sin(theta)+y*np.cos(theta)
    freq=rng.uniform(4,15);a=np.zeros((size,size));r=np.hypot(x,y)
    if family=='solid': a=(((u/.7)**2+(v/rng.uniform(.2,.7))**2)<1).astype(float)
    elif family in ['islands','blobs']:
        for _ in range(int(rng.integers(4,22))):
            cx,cy=rng.uniform(-.85,.85,2);s=rng.uniform(.03,.18)
            a=np.maximum(a,np.exp(-((x-cx)**2+(y-cy)**2)/(2*s*s)))
        if family=='islands':a=(a>.5).astype(float)
    elif family=='rings':
        for _ in range(int(rng.integers(1,6))):
            cx,cy=rng.uniform(-.6,.6,2);rr=rng.uniform(.1,.28);d=np.hypot(x-cx,y-cy)
            a=np.maximum(a,(abs(d-rr)<rng.uniform(.02,.06)).astype(float))
    elif family=='perforated':
        a=(r<.85).astype(float)
        for _ in range(int(rng.integers(4,20))):
            cx,cy=rng.uniform(-.6,.6,2);a[np.hypot(x-cx,y-cy)<rng.uniform(.04,.11)]=0
    elif family=='nested_rings':a=((np.cos(r*freq*np.pi)>rng.uniform(.0,.6))&(r<.9)).astype(float)
    elif family in ['tree','network','fractures','glyphs','scene']:
        im=Image.new('L',(size,size));d=ImageDraw.Draw(im)
        pt=lambda z:tuple(int(t*size) for t in z)
        if family=='tree':
            def branch(px,py,length,ang,depth):
                qx=px+length*np.cos(ang);qy=py+length*np.sin(ang)
                d.line([pt((px,py)),pt((qx,qy))],fill=255,width=max(1,depth))
                if depth>0:
                    for sign in [-1,1]:branch(qx,qy,length*.68,ang+sign*rng.uniform(.3,.7),depth-1)
            branch(.5,.92,.25,-np.pi/2,4)
        elif family=='network':
            points=rng.uniform(.08,.92,(int(rng.integers(12,25)),2))
            distances=np.linalg.norm(points[:,None]-points[None,:],axis=2)
            for i,p in enumerate(points):
                for j in np.argsort(distances[i])[1:4]:d.line([pt(p),pt(points[j])],fill=255,width=int(rng.integers(1,4)))
        elif family=='fractures':
            for _ in range(8):
                points=np.cumsum(rng.normal(0,.09,(15,2)),axis=0)+rng.uniform(.2,.8,2)
                d.line([pt(p) for p in points],fill=255,width=int(rng.integers(1,3)))
        elif family=='glyphs':
            for row in range(4):
                for col in range(4):
                    d.text((col*size//4,row*size//4),str(rng.choice(list('AB08OXY123'))),fill=255)
        else:
            d.rectangle((0,int(size*.65),size,size),fill=90)
            for _ in range(8):
                left=int(rng.uniform(.02,.85)*size);w=int(rng.uniform(.05,.2)*size);top=int(rng.uniform(.2,.65)*size)
                d.rectangle((left,top,left+w,int(size*.65)),fill=int(rng.integers(100,255)))
            d.ellipse((int(.1*size),int(.05*size),int(.3*size),int(.25*size)),fill=220)
        a=np.asarray(im,float)/255
    elif family=='labyrinth':a=(gaussian_filter(rng.normal(size=(size,size)),rng.uniform(1.5,3.5))>0).astype(float)
    elif family=='stripes':a=(np.sin(freq*np.pi*u+2*np.sin(3*v))>0).astype(float)
    elif family=='checkerboard':a=((np.sin(freq*np.pi*u)*np.sin(freq*np.pi*v))>0).astype(float)
    elif family=='radial':a=((np.cos(rng.integers(3,15)*np.arctan2(y,x))>0)&(r<.9)).astype(float)
    elif family=='spiral':a=((np.cos(r*freq*np.pi+int(rng.integers(2,6))*np.arctan2(y,x))>0)&(r<.9)).astype(float)
    elif family=='multiscale':
        for s in [1,2,4,8,16]:a+=gaussian_filter(rng.normal(size=(size,size)),s)*s**.7
        a=(a-a.min())/(np.ptp(a)+1e-15)
    elif family=='gradient':a=np.clip(.5+.25*u+.25*np.sin(3*v),0,1)
    elif family=='edges':a=((u>.1)&(v>-.6)&(v<.6)).astype(float);a+=.4*np.exp(-((u+.35)/.07)**2);a=np.clip(a,0,1)
    elif family=='grains':
        points=rng.uniform(-1,1,(int(rng.integers(10,35)),2));dist=(x[...,None]-points[:,0])**2+(y[...,None]-points[:,1])**2
        a=rng.uniform(0,1,len(points))[np.argmin(dist,axis=2)]
    else:raise ValueError(family)
    # Vary contrast and polarity while retaining named morphology before inversion.
    if seed%3==0:a=1-a
    a=.05+.95*a
    return np.clip(a,0,1)

def topology(binary):
    # 4-connected foreground and dual 8-connected background; padded exterior.
    _,components=label(binary,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    _,background=label(~np.pad(binary,1,constant_values=False),structure=np.ones((3,3)))
    holes=max(0,background-1)
    return components,holes,components-holes

def features(a):
    result={}
    for t in [.2,.5,.8]:
        c,h,e=topology(a>t);result.update({f'components_{t}':c,f'holes_{t}':h,f'euler_{t}':e,f'area_{t}':float(np.mean(a>t))})
    gy,gx=np.gradient(a);power=abs(np.fft.fftshift(np.fft.fft2(a-a.mean())))**2
    yy,xx=np.indices(a.shape);rr=np.hypot(yy-a.shape[0]//2,xx-a.shape[1]//2);total=power.sum()+1e-15
    hist=np.histogram(a,bins=32,range=(0,1))[0].astype(float);hist/=hist.sum();hist=hist[hist>0]
    result.update(mean=float(a.mean()),std=float(a.std()),entropy=float(-(hist*np.log2(hist)).sum()),edge_energy=float(np.mean(gx*gx+gy*gy)),high_frequency=float(power[rr>a.shape[0]/4].sum()/total),anisotropy=float(abs(np.mean(gx*gx)-np.mean(gy*gy))/(np.mean(gx*gx+gy*gy)+1e-15)))
    return result

def generate(root,count=24,size=96,variants=8,seed=2026):
    if count<len(FAMILIES) or count>len(FAMILIES)*variants:raise ValueError(f'count must be {len(FAMILIES)}..{len(FAMILIES)*variants}')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    spec=dict(count=count,size=size,variants=variants,seed=seed,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest=root/'dataset.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing['spec']!=spec:raise ValueError('Dataset parameters changed: choose a fresh output directory')
        for row in existing['selected']:
            f=root/row['file']
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Generated dataset changed or file missing')
        return [str((root/r['file']).resolve()) for r in existing['selected']]
    records=[];arrays=[]
    for fi,family in enumerate(FAMILIES):
        for v in range(variants):
            s=seed+fi*1000+v;a=make(family,s,size)
            # Features measured on actual saved 8-bit amplitudes.
            a=np.rint(a*255).astype(np.uint8)/255
            records.append(dict(family=family,variant=v,seed=s,**features(a)));arrays.append(a)
    names=[k for k in records[0] if k not in ['family','variant','seed']]
    x=np.array([[r[k] for k in names] for r in records]);x=np.sign(x)*np.log1p(abs(x))
    x=(x-np.median(x,axis=0))/(np.std(x,axis=0)+1e-8)
    # One representative per family first, then global farthest-point selection.
    selected=[]
    for family in FAMILIES:
        ids=[i for i,r in enumerate(records) if r['family']==family]
        centre=x[ids].mean(axis=0);selected.append(min(ids,key=lambda i:np.sum((x[i]-centre)**2)))
    while len(selected)<count:
        dist=((x[:,None,:]-x[selected][None,:,:])**2).sum(axis=2).min(axis=1);dist[selected]=-1
        selected.append(int(np.argmax(dist)))
    chosen=[]
    for rank,i in enumerate(selected):
        row=records[i].copy();name=f'{rank:03d}_{row["family"]}_v{row["variant"]}.png'
        Image.fromarray(np.rint(arrays[i]*255).astype(np.uint8)).save(root/name)
        row.update(file=name,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest());chosen.append(row)
    for filename,rows in [('candidate_features.csv',records),('selected_features.csv',chosen)]:
        with open(root/filename,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest.write_text(json.dumps(dict(spec=spec,selected=chosen),indent=2))
    contact_sheet(root,chosen)
    return [str((root/r['file']).resolve()) for r in chosen]

def contact_sheet(root,rows):
    from PIL import ImageFont
    cols=4;cell=180;im=Image.new('RGB',(cols*cell,((len(rows)+cols-1)//cols)*(cell+28)),'white');d=ImageDraw.Draw(im)
    for i,r in enumerate(rows):
        x=(i%cols)*cell;y=(i//cols)*(cell+28)
        a=Image.open(root/r['file']).convert('RGB').resize((cell,cell))
        im.paste(a,(x,y));d.text((x+2,y+cell+3),r['file'][:-4],fill='black')
    im.save(root/'contact_sheet.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='auto_images');p.add_argument('--count',type=int,default=24);p.add_argument('--size',type=int,default=96);p.add_argument('--variants',type=int,default=8);a=p.parse_args()
    print(f'Generated {len(generate(a.output,a.count,a.size,a.variants))} images')
