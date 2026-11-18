import io,copy
from flask import request,jsonify,render_template,send_file
from image_lab.validation import Invalid
from .storage import Storage
from .core import decode,demo,create,edit,history,summary,png,svg
from .morph import morph_paths,add_morph_path

def install(app,root):
    store=Storage(root);app.extensions['tracedesk']=store
    def body():
        b=request.get_json(silent=True)
        if not isinstance(b,dict):raise Invalid('请求需为JSON对象')
        return b
    def result(p):return jsonify(drawing=p,summary=summary(p))
    @app.get('/')
    def page():return render_template('tracedesk.html')
    @app.get('/tracedesk/state')
    def state():return jsonify(store.state())
    @app.post('/tracedesk/demo')
    def sample():return jsonify(image=store.save_image('合成线稿',demo()))
    @app.post('/tracedesk/upload')
    def upload():
        f=request.files.get('file')
        if not f:raise Invalid('请选择图片')
        return jsonify(image=store.save_image(f.filename,decode(f.read())))
    @app.get('/tracedesk/images/<key>.png')
    def image(key):return send_file(io.BytesIO(png(store.image(key)[1])),mimetype='image/png')
    @app.post('/tracedesk/drawings')
    def new():
        b=body();return result(store.save(create(b.get('name','新图稿'),b.get('width',256),b.get('height',192)),dict(manual=True)))
    @app.get('/tracedesk/drawings/<key>')
    def get(key):return result(store.layout(key))
    @app.post('/tracedesk/drawings/<key>/edit')
    def change(key):
        b=body();im=store.image(b.get('image_id')) if b.get('action')=='image' else None
        return result(store.change(key,b.get('revision'),lambda p:edit(p,b,im)))
    @app.post('/tracedesk/drawings/<key>/history')
    def hist(key):
        b=body();return result(store.change(key,b.get('revision'),lambda p:history(p,b.get('action'))))
    @app.post('/tracedesk/drawings/<key>/morph/preview')
    def morph_preview(key):
        b=body();p=store.layout(key)
        return jsonify(morph=morph_paths(p,b.get('old_id'),b.get('new_id')))
    @app.post('/tracedesk/drawings/<key>/morph/add')
    def morph_add(key):
        b=body()
        return result(store.change(key,b.get('revision'),lambda p:add_morph_path(p,b)))
    @app.post('/tracedesk/drawings/<key>/duplicate')
    def duplicate(key):
        p=store.layout(key);p['name']+=' 副本';p['undo']=[];p['redo']=[];return result(store.save(p,dict(drawing_id=key,revision=p['revision'])))
    @app.post('/tracedesk/import')
    def restore():return result(store.save(copy.deepcopy(body().get('drawing')),dict(imported=True)))
    @app.get('/tracedesk/drawings/<key>.png')
    def image_export(key):return send_file(io.BytesIO(store.render(key,request.args.get('source')=='1')),mimetype='image/png',download_name='drawing.png',as_attachment=request.args.get('download')=='1')
    @app.get('/tracedesk/drawings/<key>.svg')
    def vector(key):return send_file(io.BytesIO(svg(store.layout(key))),mimetype='image/svg+xml',download_name='drawing.svg',as_attachment=True)
    @app.get('/tracedesk/drawings/<key>.zip')
    def archive(key):return send_file(io.BytesIO(store.export(key)),mimetype='application/zip',download_name='tracedesk.zip',as_attachment=True)
