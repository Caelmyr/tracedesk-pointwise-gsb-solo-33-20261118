import io,json,os,threading,uuid,hashlib,zipfile,copy
from pathlib import Path
import numpy as np
from image_lab.validation import Invalid,Conflict
from .core import png,validate,summary,render,svg

class Storage:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.document=self.root/'workspace.json'
        if not self.document.exists():self.write(dict(revision=0,images=[],layouts=[]))
    def read(self):return json.loads(self.document.read_text())
    def write(self,d):
        tmp=self.document.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,self.document)
    def path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise Invalid('无效ID')
        return self.root/(key+'.npy')
    def image(self,key):
        with self.lock:
            row=next((x for x in self.read()['images'] if x['id']==key),None)
            if row is None:raise Invalid('没有此图片')
            raw=self.path(key).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise Invalid('图片校验失败')
            return row,np.load(io.BytesIO(raw),allow_pickle=False)
    def save_image(self,name,a):
        out=io.BytesIO();np.save(out,a,allow_pickle=False);raw=out.getvalue();key=uuid.uuid4().hex;row=dict(id=key,name=str(name)[:100],width=a.shape[1],height=a.shape[0],sha256=hashlib.sha256(raw).hexdigest())
        with self.lock:
            d=self.read();self.path(key).write_bytes(raw);d['images'].append(row);d['revision']+=1;self.write(d)
        return row
    def layout(self,key):
        with self.lock:
            row=next((x for x in self.read()['layouts'] if x['id']==key),None)
            if row is None:raise Invalid('没有此排版')
            return copy.deepcopy(validate(row))
    def save(self,p,source):
        p=copy.deepcopy(validate(p));p.update(id=uuid.uuid4().hex,revision=0,source=source)
        if p['image_id']:self.image(p['image_id'])
        with self.lock:
            d=self.read();d['layouts'].append(p);d['revision']+=1;self.write(d)
        return p
    def change(self,key,revision,callback):
        with self.lock:
            d=self.read();p=next((x for x in d['layouts'] if x['id']==key),None)
            if p is None:raise Invalid('没有此排版')
            if revision!=p['revision']:raise Conflict('排版已改变，请刷新')
            callback(p);validate(p);p['revision']+=1;d['revision']+=1;self.write(d);return copy.deepcopy(p)
    def state(self):
        with self.lock:
            d=self.read();d['layouts']=[{k:v for k,v in p.items() if k not in ('undo','redo')}|dict(summary=summary(p)) for p in d['layouts']];return d
    def render(self,key,source=False):
        p=self.layout(key);image=self.image(p['image_id'])[1] if p['image_id'] else None;return png(render(p,image,source))
    def export(self,key):
        p=self.layout(key);out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('drawing.json',json.dumps(p,ensure_ascii=False,indent=2));z.writestr('summary.json',json.dumps(summary(p),ensure_ascii=False,indent=2));z.writestr('drawing.png',self.render(key));z.writestr('drawing.svg',svg(p))
            if p['image_id']:z.writestr('original.png',png(self.image(p['image_id'])[1]))
            z.writestr('README.txt','TraceDesk手工单线矢量图稿。drawing.svg与drawing.png仅含路径，原底图独立保留。drawing.json可在素材仍存在的同工作区恢复。画布坐标为原图像素坐标，闭环使用闭合路径。')
        return out.getvalue()
