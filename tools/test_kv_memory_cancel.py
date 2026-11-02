"""Real HTTP queue ownership and cancellation/drain/reset contract tests."""
import argparse,json,sys,tempfile,threading,queue,time,urllib.request,contextlib
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ap=argparse.ArgumentParser();ap.add_argument('--runtime');ap.add_argument('--expect-race',action='store_true');a=ap.parse_args()
sys.path.insert(0,a.runtime or str(Path(__file__).resolve().parents[1]/'serve/kv_memory'));import kv_live as kv
class Tok:
 def encode(self,x,**kw):return [{'<|im_start|>':1,'<|image_pad|>':2,'<|vision_start|>':3}[x]]
class Pipe:
 def __init__(self,q):self.q=q;self.owner=None;self.ops=[];self.live=0;self.fail_reset=False
 def flush(self):pass
 def write(self,s):
  words=s.strip().split(' ');op=words[0];self.ops.append(op)
  if op=='MEM_BEGIN':self.owner=words[1]
  elif op=='MEM_END':assert words[1]==self.owner;self.owner=None
  elif op=='MEM_CALL':
   assert words[1]==self.owner;op=words[2];self.ops.append(op)
   if op=='GEN':self.live=100;return len(s)
   if op=='QBHOTRESET' and not self.fail_reset:self.live=0
   self.q.put('QBRESULT '+json.dumps({'op':op,'ok':not(op=='QBHOTRESET' and self.fail_reset),'weights_retained':True,'guard':{'ok':True},'live_hash':'x','live_count':self.live,'live_ok':bool(self.live)}));return len(s)
  elif op=='STOP':return len(s)
  else:assert op=='MEM_CAPS'
  self.q.put('MEMRESULT '+json.dumps({'version':1,'ok':True,'epoch':1,'geometry_replay':True,'mtp_active_audit':True,'transaction_fence':True}));return len(s)
with tempfile.TemporaryDirectory() as tmp:
 q=queue.Queue();pipe=Pipe(q);proc=SimpleNamespace(pid=13,stdin=pipe,poll=lambda:None)
 engine=SimpleNamespace(proc=proc,lines=q,ended=False,memory_lifecycle={'proc':proc,'run':'test','cause':'native_exit_or_external_termination','eof':False},last={})
 engine.alive=lambda:engine.proc is not None
 def close():engine.proc=None
 engine.close=close
 class S:
  tok=Tok();api_key='TEST_PRIVATE_KEY'
  def __init__(self):self.engine=engine
  def run(self,*args,**kw):
   self.engine.proc.stdin.write('GEN 1\n')
   try:yield ('content','one')
   finally:pipe.write('STOP\n')
 svc=S();journal=kv.Journal(Path(tmp)/'logs');kv._configured['journal']=journal
 started=threading.Event();release=threading.Event();observed=[]
 def make(s):
  class H(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def _json(self,status,body):self.send_response(status);self.end_headers();self.wfile.write(json.dumps(body).encode())
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    if body['label']=='A':
     s.engine.proc.stdin.write('GEN 1\n');started.set();assert release.wait(5)
     observed.append({'actual':s.engine.memory_lifecycle.get('request_id'),'expected':self.kv_request_id})
    self._json(200,{'id':self.kv_request_id})
  return H
 server=SimpleNamespace(make_handler=make);c=kv.install(svc,{'root':tmp,'namespace':'cancel','disk_budget_bytes':8*1024**3,'port':0},server)
 http=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(svc));threading.Thread(target=http.serve_forever,daemon=True).start();replies=[]
 def ask(label):
  req=urllib.request.Request(f'http://127.0.0.1:{http.server_port}/v1/chat/completions',data=json.dumps({'label':label}).encode())
  with urllib.request.urlopen(req,timeout=8) as r:replies.append(json.load(r))
 first=threading.Thread(target=ask,args=('A',));first.start();assert started.wait(5)
 second=threading.Thread(target=ask,args=('B',));second.start()
 for _ in range(200):
  if len([x for x in journal.last if x['event']=='http_start'])==2:break
  time.sleep(.005)
 else:raise AssertionError('B did not queue')
 time.sleep(.03);release.set();first.join(8);second.join(8);assert len(replies)==2
 matched=observed[0]['actual']==observed[0]['expected']
 assert matched != a.expect_race,observed
 result={'HTTP_queue_owner_stable':matched,'baseline_race_reproduced':a.expect_race}
 if not a.expect_race:
  c.audit=lambda:{'guard':{'ok':True},'live_hash':'x'};c.save=lambda *args:None
  cancel=threading.Event();g=svc.run([1,10,1,11,1,12],True,[],8192,{},cancel)
  assert next(g)==('content','one');g.close()
  assert cancel.is_set() and engine.proc is proc and pipe.live==0
  assert pipe.ops.index('STOP')<pipe.ops.index('QBHOTRESET')
  assert any(x['event']=='cancel_reset_verified' for x in journal.last)
  # A failed reset must discard this backend, never publish or keep a proof.
  pipe.fail_reset=True;cancel=threading.Event();g=svc.run([1,10,1,11,1,12],True,[],8192,{},cancel);next(g);g.close()
  assert engine.proc is None and c.warm_text is None
  assert any(x['event']=='cancel_backend_discarded' for x in journal.last)
  result.update(STOP_drained_before_reset=True,weights_retained_on_cancel=True,reset_failure_discards=True)
 http.shutdown();http.server_close();c.httpd.shutdown();c.httpd.server_close()
 print(json.dumps({'passed':True,**result,'GPU_calls':0}))
