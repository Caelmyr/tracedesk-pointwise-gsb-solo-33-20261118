import os
from image_lab import create_app
from tracedesk.server import install
from waitress import serve
root=os.environ.get('TRACE_DATA_DIR','data')
app=create_app(os.path.join(root,'preparation'))
install(app,os.path.join(root,'drawings'))
if __name__=='__main__':serve(app,host='0.0.0.0',port=int(os.environ.get('PORT','18936')),threads=4)
