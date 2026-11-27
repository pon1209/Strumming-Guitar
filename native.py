import sys, pathlib, os, json, threading, traceback, math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT=pathlib.Path(getattr(sys,'_MEIPASS',pathlib.Path(__file__).resolve().parent))
LOCK=threading.Lock(); MODEL=None

def analyze(x,sr):
    global MODEL
    import numpy as np, librosa, torch
    from klangio.modules.strumming_crnn import HighResStrummingTracking
    torch.set_num_threads(2)
    if not np.isfinite(x).all() or len(x)<sr//2: raise ValueError('Audio inválido o demasiado corto')
    if MODEL is None:
        MODEL=HighResStrummingTracking(str(ROOT/'checkpoint.ckpt'),device='cpu')
    tempo,beats=librosa.beat.beat_track(y=x,sr=sr)
    bpm=float(np.asarray(tempo).reshape(-1)[0]); offset=float(librosa.frames_to_time(beats[0],sr=sr)) if len(beats) else 0
    events=[]
    # Overlap context, retain only central region to limit memory and duplicate events.
    for a in range(0,len(x),sr*20):
        b=min(len(x),a+sr*20); lo=max(0,a-sr*2); hi=min(len(x),b+sr*2)
        chunk=x[lo:hi]; chunk=np.pad(chunk,(0,max(0,sr*3-len(chunk))))
        for e in MODEL.process(chunk,sr):
            t=float(e.time)+lo/sr
            if a/sr<=t<b/sr and math.isfinite(t):
                ch=str(e.chord).replace(':maj','').replace(':min','m')
                events.append({'time':t,'direction':{'D':'↓','U':'↑'}.get(e.action,'?'),'chord':ch,'method':'Klangio CRNN'})
    events.sort(key=lambda e:e['time'])
    return {'events':events,'bpm':bpm if 30<=bpm<=300 else None,'offset':offset}

class Handler(BaseHTTPRequestHandler):
    def send_json(self,obj,code=200):
        b=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode(); self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p=self.path.split('?')[0]; p='/index.html' if p=='/' else p
        if p not in ('/index.html','/finger.js','/score.js','/fusion.js','/pitch.bundle.js') and not p.startswith('/model/'): self.send_error(404); return
        f=(ROOT/p.lstrip('/')).resolve()
        if not f.is_relative_to(ROOT.resolve()) or not f.is_file(): self.send_error(404); return
        b=f.read_bytes(); self.send_response(200); self.send_header('Content-Type',{'.html':'text/html; charset=utf-8','.js':'application/javascript','.json':'application/json'}.get(f.suffix,'application/octet-stream')); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        if self.path!='/klangio' or self.headers.get('Origin')!=f'http://127.0.0.1:{self.server.server_port}' or self.headers.get('Content-Type')!='application/octet-stream': self.send_error(403); return
        if not LOCK.acquire(False): self.send_json({'error':'Motor ocupado. Espera a que termine el análisis anterior.'},409); return
        try:
            import numpy as np
            n=int(self.headers.get('Content-Length','0'))
            if n<=0 or n%4 or n>22050*4*600: raise ValueError('Máximo 10 minutos de audio')
            x=np.frombuffer(self.rfile.read(n),dtype='<f4').copy()
            result=analyze(x,22050)
            self.send_json(result)
        except Exception as e:
            traceback.print_exc()
            try:self.send_json({'error':str(e)},500)
            except (BrokenPipeError,ConnectionResetError):pass
        finally:LOCK.release()

def main():
    if len(sys.argv)>2 and sys.argv[1]=='--self-test':
        import librosa
        x,sr=librosa.load(sys.argv[2],sr=22050,duration=12)
        r=analyze(x,sr); print(json.dumps(r,ensure_ascii=False)); assert len(r['events'])>0,'Klangio no detectó rasgueos en la muestra oficial'; return
    import webview
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler); server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    webview.create_window('Strumming Studio 9 · Precision',f'http://127.0.0.1:{server.server_port}',width=1200,height=850,min_size=(650,550))
    try:webview.start(gui='cocoa')
    finally:server.shutdown()
if __name__=='__main__':main()
