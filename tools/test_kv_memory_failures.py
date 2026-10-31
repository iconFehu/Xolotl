"""Real pipes/HTTP/SSE tests, no model or credentials needed."""
import sys, subprocess, queue, threading, tempfile, time, json, urllib.request, socket
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serve/kv_memory'))
import kv_live as kv
from observability import engine_class, snapshot, instrument_cancel

with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);j=kv.Journal(root/'logs');kv._configured['journal']=j;checks=[]
 class Base:
  def close(self):
   if self.proc and self.proc.poll() is None:self.proc.terminate();self.proc.wait(timeout=3)
   self.proc=None
 E=engine_class(Base,j,('ERR verify: timed out at layer ',))
 def spawn(code):
  e=E();e.memory_run_id=__import__('uuid').uuid4().hex;e.lines=queue.Queue();e.slot_q=[]
  e.proc=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
  e.ended=False;t=threading.Thread(target=e._pump);t.start();return e,t
 e,t=spawn('import time,sys;time.sleep(.1);sys.exit(37)');t.join(5);assert not t.is_alive()
 assert e.lines.get(timeout=2) is None and snapshot(e,wait=1)['exit_code']==37
 time.sleep(.05);assert any(x['event']=='native_exit' and x['exit_code']==37 for x in j.last);checks.append('real_process_exit_code_37')
 e,t=spawn("import time;time.sleep(.1);print('ERR verify: timed out at layer 5 SECRET_SENTINEL',flush=True);time.sleep(20)")
 t.join(5);assert not t.is_alive();assert snapshot(e,wait=1)['fatal_code']=='VERIFY_TIMEOUT';checks.append('fatal_stdout_not_lost')
 assert 'SECRET_SENTINEL' not in (root/'logs/events.jsonl').read_text()
 e.close()
 cancel=instrument_cancel(threading.Event(),j,'cancel-test');cancel._kv_reason='client_disconnected';cancel.set();cancel.set()
 assert len([x for x in j.last if x['event']=='request_cancelled' and x['request_id']=='cancel-test'])==1;checks.append('cancel_once_with_reason')

 # Build a controller with a real process/queue EOF, then a fresh subprocess.
 class Tok:
  def encode(self,x,**kw):return [{'<|im_start|>':1,'<|image_pad|>':2,'<|vision_start|>':3}[x]]
 class Service:
  tok=Tok();api_key='TEST_SECRET_NOT_LOGGED'
  def __init__(self):self.engine=SimpleNamespace(proc=None,alive=lambda:True,close=lambda:None,last={})
  def run(self,*a,**kw):yield ('content','unused')
 service=Service();c=kv.Controller(service,{'root':tmp,'namespace':'faults','disk_budget_bytes':8*1024**3})
 ids=[1,10,1,11,1,12];c.retention=lambda:None;c.save=lambda *a:None;c.gate=__import__('contextlib').nullcontext()
 c.qb=lambda *a,**kw:{'guard':{},'live_hash':'ok','live_ok':True,'live_count':100}
 audits=[]
 def audit():
  audits.append(1)
  if len(audits)==1:raise RuntimeError('KV_CONTROL_EOF')
  return {'guard':{},'live_hash':'ok'}
 c.audit=audit;attempts=[]
 def generation(*a,**kw):
  attempts.append(1);yield ('ping',None);yield ('content','UNCHECKED' if len(attempts)==1 else 'VERIFIED_COLD')
 c.original_run=generation;service.engine.restart=lambda:None
 def warm():c.warm_text={'proc':service.engine.proc,'prefix_ids':ids[:4],'live_count':10,'check':{'guard':{},'live_hash':'ok'}}
 warm();out=list(c.run(ids,True,[],8192,{},threading.Event()))
 assert [x for x in out if x[0]!='ping']==[('content','VERIFIED_COLD')] and len(attempts)==2 and len(audits)==2
 checks.append('buffered_answer_discarded_retry_audited_once')
 def failed_audit():raise RuntimeError('MTP_ACTIVE_VALIDITY')
 c.audit=failed_audit;warm();attempts.clear()
 out=[]
 try:
  for x in c.run(ids,True,[],8192,{},threading.Event()):out.append(x)
  raise AssertionError('must fail closed')
 except RuntimeError:pass
 assert len(attempts)==2 and all(x[0]=='ping' for x in out);checks.append('retry_failure_no_model_output')
 cancel=threading.Event()
 def cancelled(*a,**kw):yield ('ping',None);cancel.set();yield ('content','CANCELLED_BUFFER')
 c.original_run=cancelled;warm();out=list(c.run(ids,True,[],8192,{},cancel));assert out==[('ping',None)]
 checks.append('cancel_skips_audit_no_retry_no_buffer_release')
 # Cold output already delivered must never be replayed after audit failure.
 c.warm_text=None;attempts.clear();c.original_run=generation
 try:list(c.run(ids,True,[],8192,{},threading.Event()))
 except RuntimeError:pass
 assert len(attempts)==1;checks.append('published_output_never_replayed')

 # Actual HTTP: headers=200 must carry an application error and a proper end,
 # and server logs must mark it failed. Each supported SSE dialect is checked.
 def make(s):
  class H(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def _json(self,status,body):self.send_response(status);self.end_headers();self.wfile.write(json.dumps(body).encode())
   def _sse(self):self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   def do_POST(self):self.rfile.read(int(self.headers['Content-Length']));self._sse();raise RuntimeError('KV_CONTROL_EOF:TEST_SECRET_NOT_LOGGED')
  return H
 (root/'http').mkdir()
 server=SimpleNamespace(make_handler=make);service=Service();control=kv.install(service,{'root':str(root/'http'),'namespace':'http','disk_budget_bytes':8*1024**3,'port':0},server)
 http=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(service));threading.Thread(target=http.serve_forever,daemon=True).start()
 for path in ('/v1/chat/completions','/v1/responses','/v1/messages'):
  with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{http.server_port}'+path,data=b'{}'),timeout=5) as r:
   body=r.read().decode();assert r.status==200 and 'KV_REQUEST_FAILED' in body and r.headers['X-KV-Request-ID'] in body
   if path.endswith('completions'):assert 'data: [DONE]' in body
   elif path.endswith('responses'):assert 'event: response.failed' in body
   else:assert 'event: error' in body
 assert len([x for x in j.last if x['event']=='http_end' and x['outcome']=='failed'])==3
 checks.append('three_real_HTTP_SSE_error_dialects')
 assert 'TEST_SECRET_NOT_LOGGED' not in (root/'logs/events.jsonl').read_text()
 http.shutdown();http.server_close();control.httpd.shutdown();control.httpd.server_close()
 print(json.dumps({'passed':True,'checks':checks,'GPU_calls':0}))
