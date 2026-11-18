import io,copy,math,html
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from image_lab.validation import Invalid

def integer(v,lo,hi,name):
    if isinstance(v,bool):raise Invalid(name+'需要整数')
    try:n=int(v)
    except (TypeError,ValueError):raise Invalid(name+'需要整数')
    if n!=float(v) or not lo<=n<=hi:raise Invalid(name+'超出范围')
    return n

def number(v,lo,hi,name):
    try:n=float(v)
    except (TypeError,ValueError):raise Invalid(name+'需要数值')
    if not math.isfinite(n) or not lo<=n<=hi:raise Invalid(name+'超出范围')
    return n

def png(a):
    out=io.BytesIO();Image.fromarray(np.asarray(a,dtype=np.uint8),'RGBA').save(out,format='PNG');return out.getvalue()

def decode(raw):
    if len(raw)>12*1024*1024:raise Invalid('文件超过12MB')
    try:
        im=Image.open(io.BytesIO(raw));im.load()
        if max(im.size)>2048:raise Invalid('图片最长边不能超过2048')
        return np.array(ImageOps.exif_transpose(im).convert('RGBA'))
    except Invalid:raise
    except Exception:raise Invalid('无法读取图片')

def demo():
    im=Image.new('RGBA',(256,192),'white');d=ImageDraw.Draw(im)
    d.line([(25,145),(80,75),(125,135),(175,45),(230,150)],fill='#222222',width=7,joint='curve');d.line([(80,75),(85,30)],fill='#222222',width=5);d.ellipse((135,85,195,145),outline='#222222',width=7)
    d.text((5,175),'SYNTHETIC LINE ART',fill='#777777');return np.array(im)

def create(name,w=256,h=192):return dict(name=str(name)[:100],width=integer(w,32,2048,'画布宽'),height=integer(h,32,2048,'画布高'),image_id=None,image_opacity=.35,stroke='#256b72',line_width=2,paths=[],undo=[],redo=[])
def snapshot(p):return {k:copy.deepcopy(v) for k,v in p.items() if k not in ('id','revision','source','undo','redo')}
def checkpoint(p):p['undo']=(p.get('undo',[])+[snapshot(p)])[-30:];p['redo']=[]

def validate(p):
    if not isinstance(p,dict):raise Invalid('图稿需为对象')
    w=integer(p.get('width'),32,2048,'画布宽');h=integer(p.get('height'),32,2048,'画布高');number(p.get('image_opacity'),0,1,'底图透明度');number(p.get('line_width'),.2,20,'线宽')
    if not isinstance(p.get('paths'),list) or len(p['paths'])>500:raise Invalid('最多500条路径')
    if not isinstance(p.get('stroke'),str) or len(p['stroke'])!=7 or p['stroke'][0]!='#':raise Invalid('线色无效')
    try:int(p['stroke'][1:],16)
    except ValueError:raise Invalid('线色无效')
    ids=set()
    for q in p['paths']:
        if not isinstance(q,dict) or not isinstance(q.get('id'),str) or q['id'] in ids:raise Invalid('路径ID无效或重复')
        ids.add(q['id'])
        if not isinstance(q.get('points'),list) or not 2<=len(q['points'])<=4096:raise Invalid('每条路径需2到4096点')
        for a in q['points']:
            if not isinstance(a,list) or len(a)!=2:raise Invalid('坐标需为二维')
            number(a[0],0,w,'横坐标');number(a[1],0,h,'纵坐标')
        if not isinstance(q.get('closed',False),bool):raise Invalid('闭合值无效')
    return p

def summary(p):
    length=0
    for q in p['paths']:
        points=q['points']+([q['points'][0]] if q.get('closed') else [])
        length+=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(points,points[1:]))
    return dict(path_count=len(p['paths']),node_count=sum(len(q['points']) for q in p['paths']),closed_count=sum(q.get('closed',False) for q in p['paths']),length=round(length,2),image_id=p['image_id'])

def edit(p,b,image=None):
    import uuid
    action=b.get('action');checkpoint(p)
    if action=='settings':
        if 'name' in b:p['name']=str(b['name'])[:100]
        if 'stroke' in b:p['stroke']=str(b['stroke'])
        for k,lo,hi in [('image_opacity',0,1),('line_width',.2,20)]:
            if k in b:p[k]=number(b[k],lo,hi,k)
    elif action=='image':
        if image is None:raise Invalid('请选择底图')
        row,a=image
        if p['paths'] and (p['width']!=a.shape[1] or p['height']!=a.shape[0]):raise Invalid('已有路径时请先清空或新建同尺寸图稿')
        p.update(image_id=row['id'],width=a.shape[1],height=a.shape[0])
    elif action=='add':
        p['paths'].append(dict(id=uuid.uuid4().hex,name=str(b.get('name','手绘路径'))[:80],points=copy.deepcopy(b.get('points')),closed=bool(b.get('closed',False))))
    elif action in ('path','delete','reverse','duplicate','move','point','delete_point'):
        q=next((q for q in p['paths'] if q['id']==b.get('path_id')),None)
        if q is None:raise Invalid('请选择路径')
        if action=='delete':p['paths'].remove(q)
        elif action=='reverse':q['points'].reverse()
        elif action=='duplicate':
            v=copy.deepcopy(q);v['id']=uuid.uuid4().hex;v['name']+=' 副本';p['paths'].append(v)
        elif action=='move':
            dx=number(b.get('dx'),-p['width'],p['width'],'横向移动');dy=number(b.get('dy'),-p['height'],p['height'],'纵向移动');q['points']=[[x+dx,y+dy] for x,y in q['points']]
        elif action in ('point','delete_point'):
            i=integer(b.get('index'),0,len(q['points'])-1,'节点序号')
            if action=='delete_point':q['points'].pop(i)
            else:q['points'][i]=[number(b.get('x'),0,p['width'],'横坐标'),number(b.get('y'),0,p['height'],'纵坐标')]
        else:
            if 'name' in b:q['name']=str(b['name'])[:80]
            if 'closed' in b:q['closed']=bool(b['closed'])
    elif action=='clear':p['paths']=[]
    else:raise Invalid('未知编辑动作')
    validate(p)

def history(p,action):
    if action not in ('undo','redo'):raise Invalid('未知历史动作')
    other='redo' if action=='undo' else 'undo'
    if not p[action]:raise Invalid('没有可恢复的历史')
    current=snapshot(p);saved=p[action].pop();p[other]=(p[other]+[current])[-30:]
    for k,v in saved.items():p[k]=v
    validate(p)

def render(p,image=None,source=False):
    out=Image.new('RGBA',(p['width'],p['height']),'white')
    if source and image is not None:
        im=Image.fromarray(image,'RGBA');alpha=im.getchannel('A').point(lambda v:round(v*p['image_opacity']));im.putalpha(alpha);out.alpha_composite(im)
    d=ImageDraw.Draw(out)
    for q in p['paths']:
        pts=[tuple(v) for v in q['points']]
        if q.get('closed'):pts.append(pts[0])
        d.line(pts,fill=p['stroke'],width=max(1,round(p['line_width'])),joint='curve')
    return np.array(out)

def svg(p):
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{p["width"]}" height="{p["height"]}" viewBox="0 0 {p["width"]} {p["height"]}">',f'<title>{html.escape(p["name"])}</title>',f'<g fill="none" stroke="{p["stroke"]}" stroke-width="{p["line_width"]}" stroke-linecap="round" stroke-linejoin="round">']
    for q in p['paths']:
        points=q['points'];d='M '+' L '.join(f'{float(x):.4f} {float(y):.4f}' for x,y in points)+(' Z' if q.get('closed') else '')
        lines.append(f'<path d="{d}" data-path-id="{q["id"]}"/>')
    lines+=['</g>','</svg>'];return '\n'.join(lines).encode()
