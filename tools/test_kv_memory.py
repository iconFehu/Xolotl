import sys,io,json,queue,tempfile,threading,urllib.request,struct
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serve/kv_memory'))
import kv_live as kv
class Pipe:
 def __init__(self,q):self.q=q;self.owner=None;self.epoch=0
 def flush(self):pass
 def write(self,s):
  a=s.strip().split(' ');op=a[0]
  if op=='MEM_BEGIN':assert not self.owner;self.owner=a[1];self.epoch+=1
  elif op=='MEM_END':assert a[1]==self.owner;self.owner=None
  elif op=='MEM_CALL':
   assert a[1]==self.owner
   if a[2].startswith('QB'):self.q.put('QBRESULT '+json.dumps({'op':a[2],'ok':True,'guard':{'gdn':'0','ple':'0','index_running':'0'},'live_hash':'0','live_ok':True,'live_count':1000}))
   return len(s)
  else:assert op=='MEM_CAPS'
  self.q.put('MEMRESULT '+json.dumps({'version':1,'ok':True,'epoch':self.epoch,'geometry_replay':True,'mtp_active_audit':True,'transaction_fence':True}));return len(s)
class Tokenizer:
 def encode(self,t,parse_special=False):return [{'<|im_start|>':1,'<|image_pad|>':2,'<|vision_start|>':3}[t]]
with tempfile.TemporaryDirectory(prefix='kv-live-cpu-') as t:
 q=queue.Queue();pipe=Pipe(q);engine=SimpleNamespace(proc=SimpleNamespace(stdin=pipe,pid=13,poll=lambda:None),lines=q,last={'reused':0,'prompt_read':6,'generated':1},alive=lambda:True,close=lambda:None)
 class Service:
  api_key='DO_NOT_LOG_THIS_CREDENTIAL';tok=Tokenizer()
  def __init__(self):self.engine=engine
  def run(self,*args,**kw):self.engine.proc.stdin.write('GEN 6 1\n');yield ('content','test answer')
 svc=Service();kv._configured['journal']=kv.Journal(Path(t)/'logs')
 def make(s):
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def _json(self,status,body):self.send_response(status);self.end_headers();self.wfile.write(json.dumps(body).encode())
   def do_POST(self):
    self.rfile.read(int(self.headers['Content-Length']));answer=list(s.run([1,10,1,11,1,12],True,[],8192,{},threading.Event()));self._json(200,{'result':answer})
  return Handler
 server=SimpleNamespace(make_handler=make);c=kv.install(svc,{'root':t,'port':0,'namespace':'test','disk_budget_bytes':8*1024**3},server)
 c.audit=lambda:{'guard':{'gdn':'0','ple':'0','index_running':'0'},'live_hash':'0'}
 httpd=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(svc));threading.Thread(target=httpd.serve_forever,daemon=True).start()
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions',data=b'{}'),timeout=5) as r:
  assert r.status==200 and len(r.headers['X-KV-Request-ID'])==32;body=json.load(r);assert body['result']==[['content','test answer']]
 rows=[json.loads(x) for x in (Path(t)/'logs/events.jsonl').read_text().splitlines()]
 assert {'http_start','http_end','generation_start','generation_done','transaction'}<={x['event'] for x in rows}
 assert 'DO_NOT_LOG_THIS_CREDENTIAL' not in (Path(t)/'logs/events.jsonl').read_text()
 try:raise ValueError(svc.api_key)
 except Exception as e:c.journal.exception('injected_error',e)
 assert svc.api_key not in (Path(t)/'logs/events.jsonl').read_text()
 assert json.loads((Path(t)/'logs/events.jsonl').read_text().splitlines()[-1])['stack']
 media_path=Path(t)/'media.bin';media_path.write_bytes(struct.pack('<5i',0x31455653,4,2,2,2)+struct.pack('<8f',*range(8)))
 svc.embeddings=SimpleNamespace(path=str(media_path),keys=['1234567890abcdef'])
 image_ids=[1,10,1,11,3,2,2,2,2,20,1,12]
 for prompt in (image_ids,image_ids,image_ids+[40,41]):list(svc.run(prompt,True,[],8192,{},threading.Event()))
 events=[json.loads(x) for x in (Path(t)/'logs/events.jsonl').read_text().splitlines()]
 assert len([x for x in events if x['event']=='image_reuse_selected'])==2
 # Changed encoder output, history, backend identity and native guard each
 # invalidate a resident image lease without publishing a reused answer.
 media_path.write_bytes(struct.pack('<5i',0x31455653,4,2,2,2)+struct.pack('<8f',*range(1,9)))
 list(svc.run(image_ids,True,[],8192,{},threading.Event()))
 changed=list(image_ids);changed[1]=999;list(svc.run(changed,True,[],8192,{},threading.Event()))
 c.warm_images['proc']=object();list(svc.run(changed,True,[],8192,{},threading.Event()))
 c.warm_images['check']={'guard':{},'live_hash':'mismatch'};list(svc.run(changed,True,[],8192,{},threading.Event()))
 events=[json.loads(x) for x in (Path(t)/'logs/events.jsonl').read_text().splitlines()]
 assert len([x for x in events if x['event']=='image_reuse_selected'])==2
 assert any(x['event']=='image_reuse_rejected' for x in events)
 text_ids=[1,10,1,11,1,12]
 for prompt in (text_ids,text_ids,text_ids+[40,41]):list(svc.run(prompt,True,[],8192,{},threading.Event()))
 events=[json.loads(x) for x in (Path(t)/'logs/events.jsonl').read_text().splitlines()]
 assert len([x for x in events if x['event']=='text_reuse_selected'])==2
 altered=list(text_ids);altered[1]=999;list(svc.run(altered,True,[],8192,{},threading.Event()))
 c.warm_text['proc']=object();list(svc.run(altered,True,[],8192,{},threading.Event()))
 c.warm_text['check']={'guard':{},'live_hash':'mismatch'};list(svc.run(altered,True,[],8192,{},threading.Event()))
 events=[json.loads(x) for x in (Path(t)/'logs/events.jsonl').read_text().splitlines()]
 assert len([x for x in events if x['event']=='text_reuse_selected'])==2
 assert any(x['event']=='text_reuse_rejected' for x in events)
 cancelled=threading.Event();cancelled.set();list(svc.run(altered,True,[],8192,{},cancelled));assert c.warm_text is None
 # An audit failure must never publish the buffered warm answer. Only the
 # subsequent complete-text retry is released, and no resident proof survives.
 list(svc.run(altered,True,[],8192,{},threading.Event()))
 old_run=c.original_run;old_audit=c.audit;passes=[];engine.restart=lambda:None
 def simulated_run(*args,**kwargs):
  passes.append(1);yield ('content','BAD_BUFFERED' if len(passes)==1 else 'GOOD_COLD_RETRY')
 audits=[]
 def rejected_audit():
  audits.append(1)
  if len(audits)==1:raise RuntimeError('MTP_ACTIVE_VALIDITY')
  return old_audit()
 c.original_run=simulated_run;c.audit=rejected_audit
 released=list(svc.run(altered,True,[],8192,{},threading.Event()))
 assert released==[('content','GOOD_COLD_RETRY')] and c.warm_text is None and len(audits)==2
 c.original_run=old_run;c.audit=old_audit
 assert pipe.owner is None
 httpd.shutdown();httpd.server_close();c.httpd.shutdown();c.httpd.server_close()
 print(json.dumps({'passed':True,'actual_HTTP':1,'response_request_id':True,'native_fence_used':True,'normal_answer_preserved':True,'structured_error_stack':True,'credential_not_logged':True,'text_history_backend_guard_invalidation':True,'cancel_invalidates':True,'bad_warm_answer_not_released':True,'GPU_calls':0}))
