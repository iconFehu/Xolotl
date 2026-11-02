"""Test-environment KV interface. Original API remains the frontend on 8080.
All model writes share one transaction; 20981 is loopback diagnostics/control.
No prompt, response, Authorization, API key or raw tool output is logged.
"""
import collections,contextlib,hashlib,hmac,importlib,json,os,queue,shutil,struct,sys,threading,time,traceback,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from memory_gate import MemoryGate
from observability import snapshot, instrument_cancel, engine_class
from native_errors import error_reference

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(1048576):h.update(chunk)
 return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def atomic(p,data):
 p=Path(p);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 with t.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def media_identity(ids,pad,path,keys):
 """Bind each actual GENI record to position, grid, encoder bytes and key.
 URLs and prompt special-token spelling are not image identities. Nothing
 from this descriptor, except aggregate hashes/counts, enters request logs.
 """
 if not path or not keys:raise RuntimeError('IMAGE_IDENTITY_UNAVAILABLE')
 starts=[i for i,x in enumerate(ids) if x==pad and (i==0 or ids[i-1]!=pad)]
 records=[]
 with Path(path).open('rb') as f:
  for start,key in zip(starts,keys):
   raw=f.read(20)
   if len(raw)!=20:raise RuntimeError('IMAGE_IDENTITY_LAYOUT')
   magic,n,nx,ny,width=struct.unpack('<5i',raw)
   if magic!=0x31455653 or n<1 or nx<1 or ny<1 or n!=nx*ny or width<1 or width>65536 or start+n>len(ids):raise RuntimeError('IMAGE_IDENTITY_LAYOUT')
   if any(x!=pad for x in ids[start:start+n]) or (start+n<len(ids) and ids[start+n]==pad):raise RuntimeError('IMAGE_IDENTITY_LAYOUT')
   if not isinstance(key,str) or len(key)!=16 or any(c not in '0123456789abcdef' for c in key):raise RuntimeError('IMAGE_IDENTITY_KEY')
   h=hashlib.sha256(raw);left=n*width*4
   while left:
    chunk=f.read(min(left,1048576))
    if not chunk:raise RuntimeError('IMAGE_IDENTITY_LAYOUT')
    h.update(chunk);left-=len(chunk)
   records.append({'start':start,'end':start+n,'grid':[n,nx,ny,width],'key':key,'sha256':h.hexdigest()})
  if len(records)!=len(keys) or len(records)!=len(starts) or f.read(1):raise RuntimeError('IMAGE_IDENTITY_LAYOUT')
 return records

def warm_image_match(warm,proc,ids,media):
 if not warm:return False,'no_verified_resident_image_state'
 if warm['proc'] is not proc:return False,'backend_changed'
 n=len(warm['prefix_ids'])
 if n>=len(ids)-1 or ids[:n]!=warm['prefix_ids']:return False,'history_prefix_changed'
 if any(x['start']<n<x['end'] for x in media):return False,'image_boundary_changed'
 # Native QSA also compares every image below the old live extent before it
 # mounts a prefix checkpoint. Reject early if a new image entered that range.
 if [x for x in media if x['start']<warm['live_count']]!=warm['media']:return False,'image_content_grid_position_or_encoding_changed'
 return True,'exact_history_media_and_backend'

def warm_text_match(warm,proc,ids):
 if not warm:return False,'no_verified_resident_text_state'
 if warm['proc'] is not proc:return False,'backend_changed'
 n=len(warm['prefix_ids'])
 if n<1 or n>=len(ids)-1 or ids[:n]!=warm['prefix_ids']:return False,'history_prefix_changed'
 return True,'exact_text_history_and_backend'

class Journal:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.local=threading.local();self.last=collections.deque(maxlen=100)
  keyfile=self.root.parent/'native-log-key.bin'
  if keyfile.is_symlink():raise RuntimeError('NATIVE_LOG_KEY_INVALID')
  if not keyfile.exists():atomic(keyfile,os.urandom(32))
  self.native_error_key=keyfile.read_bytes()
  if len(self.native_error_key)!=32:raise RuntimeError('NATIVE_LOG_KEY_INVALID')
 def emit(self,event,**data):
  row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,'request_id':getattr(self.local,'request_id',None),'pid':os.getpid(),**data}
  with self.lock:
   p=self.root/'events.jsonl'
   if p.exists() and p.stat().st_size>8*1024**2:
    for i in range(7,0,-1):
     a=self.root/f'events.{i}.jsonl';b=self.root/f'events.{i+1}.jsonl'
     if a.exists():os.replace(a,b)
    os.replace(p,self.root/'events.1.jsonl')
   with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   self.last.append(row)
 def exception(self,event,error,**data):
  code=str(error).split(':',1)[0];known={'KV_CONTROL_EOF','KV_CONTROL_REJECTED','KV_CONTROL_PROTOCOL','KV_MEMBER_INTEGRITY','RESTORE_STATE_MISMATCH','PINNED_GEOMETRY_MISSING','DISK_BUDGET_EXCEEDED','MTP_ACTIVE_VALIDITY','MEMORY_UNEXPECTED_NATIVE_RESPONSE','MEMORY_PROTOCOL_REJECTED','MEMORY_PROTOCOL_DESYNC','MEMORY_CAPABILITY_MISSING','MEMORY_BACKEND_EOF','RESIDENT_STATE_MISMATCH','RESIDENT_BACKEND_CHANGED','KV_CONTROL_TIMEOUT'}
  native=data.get('native_error');safe_code=native.get('code') if isinstance(native,dict) else None
  self.emit(event,error_type=type(error).__name__,error_code=code if code in known else safe_code,stack=[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(error.__traceback__)],**data)

_configured={}
def configure(cfg,server):
 settings=cfg.get('memory_gateway') or {}
 if not settings.get('enabled'):return None
 root=Path(settings['root']).resolve();root.mkdir(parents=True,exist_ok=True)
 journal=Journal(root/'logs');_configured.update(root=root,journal=journal)
 Base=engine_class(server.StrataEngine,journal,server.FATAL_PREFIXES)
 class Engine(Base):
  def __init__(self,*args,**kw):
   # Every restart gets a UUID; PID reuse cannot overwrite earlier diagnostics.
   args=list(args);env=dict(kw.get('env') if 'env' in kw else args[4] if len(args)>4 else {})
   ident=uuid.uuid4().hex;run=root/'native'/ident;run.mkdir(parents=True)
   env['STRATA_QSA_V2_WINDOW_TRACE']=str(run/'trace')
   if len(args)>4:args[4]=env
   else:kw['env']=env
   if len(args)>3:args[3]=str(run/'stderr.log')
   else:kw['log']=str(run/'stderr.log')
   self.memory_run_id=ident;self.memory_start_request_id=getattr(journal.local,'request_id',None)
   super().__init__(*args,**kw);self.silence_s=30;self.memory_trace=str(run/'trace')+'.pid'+str(self.proc.pid)
   journal.emit('backend_ready',engine_pid=self.proc.pid,native_run=ident,engine_sha256=digest(args[0]) if args and Path(args[0]).is_file() else None)
 server.StrataEngine=Engine
 return SimpleNamespace(root=root,journal=journal)

class Controller:
 def __init__(self,svc,settings):
  self.svc=svc;self.engine=svc.engine;self.settings=settings;self.root=Path(settings['root']);self.journal=_configured['journal'];self.gate=MemoryGate(self);self.enabled=True;self.force_cold=False;self.entries=[];self.bytes_budget=int(settings['disk_budget_bytes']);self.catalog=self.root/'catalog.json';self.min_tokens=4096;self.max_save_tokens=65536
  keyfile=self.root/'catalog-key.bin'
  if not keyfile.exists():atomic(keyfile,os.urandom(32))
  self.key=keyfile.read_bytes();self.namespace=settings['namespace'];self.im_start=svc.tok.encode('<|im_start|>',parse_special=True)[0]
  self.image_ids=set();self.warm_images=None;self.warm_text=None;self.image_pad=svc.tok.encode('<|image_pad|>',parse_special=True)[0]
  for tag in ('<|image_pad|>','<|vision_start|>'):
   encoded=svc.tok.encode(tag,parse_special=True)
   if len(encoded)==1:self.image_ids.update(encoded)
  self.original_run=svc.run;self.gate.install_service(svc);svc.run=self.run
  try:
   signed=json.loads(self.catalog.read_text());body=signed['body'];assert hmac.compare_digest(hmac.new(self.key,canonical(body),'sha256').hexdigest(),signed['mac']);assert body['namespace']==self.namespace;self.entries=body['entries']
  except FileNotFoundError:pass
  except Exception as e:self.journal.exception('catalog_rejected',e)
  class TxEvents(collections.deque):
   def append(r,event):
    super().append(event);detail=dict(event);detail['transaction_event']=detail.pop('event');self.journal.emit('transaction',**detail)
  self.gate.events=TxEvents(maxlen=2000)
 def usage(self):return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
 def qb(self,op,arg=''):
  with self.gate:
   try:self.engine.proc.stdin.write(op+(' '+arg if arg else '')+'\n');self.engine.proc.stdin.flush()
   except (OSError,ValueError):
    self.journal.emit('kv_control_failed',operation=op,phase='write',reason='KV_CONTROL_EOF',**snapshot(self.engine,wait=.25));raise RuntimeError('KV_CONTROL_EOF') from None
   deadline=time.monotonic()+120
   while True:
    try:line=self.engine.lines.get(timeout=max(.01,deadline-time.monotonic()))
    except queue.Empty:
     self.journal.emit('kv_control_failed',operation=op,reason='KV_CONTROL_TIMEOUT',**snapshot(self.engine));raise RuntimeError('KV_CONTROL_TIMEOUT') from None
    if line is None:
     self.journal.emit('kv_control_failed',operation=op,reason='KV_CONTROL_EOF',**snapshot(self.engine,wait=.25));raise RuntimeError('KV_CONTROL_EOF')
    if line.startswith('QBRESULT '):
     r=json.loads(line[9:])
     if r.get('op')!=op or not r.get('ok') or r.get('poisoned'):raise RuntimeError('KV_CONTROL_REJECTED')
     return r
    if line.startswith(('ERR ','FATAL ','MEMRESULT ')):raise RuntimeError('KV_CONTROL_PROTOCOL')
 def session(self,op,file):
  # Original production parser performs bounded reads, detects malformed
  # replies and kills a backend whose restore failed after device transfer.
  return self.engine.session_file(op,str(file))
 def verify(self,e):
  d=self.root/'blocks'/e['id'];assert d.parent==self.root/'blocks' and len(e['id'])==32
  for name,meta in e['files'].items():
   assert Path(name).name==name
   p=d/name
   if p.is_symlink() or p.stat().st_size!=meta['bytes'] or digest(p)!=meta['sha256']:raise RuntimeError('KV_MEMBER_INTEGRITY')
  return d
 def persist_catalog(self):
  body={'namespace':self.namespace,'entries':self.entries};atomic(self.catalog,canonical({'body':body,'mac':hmac.new(self.key,canonical(body),'sha256').hexdigest()}))
 def choose(self,ids):
  candidates=[e for e in self.entries if e['tokens']<len(ids)-1 and ids[:e['tokens']]==e['prefix_ids']]
  return max(candidates,key=lambda e:e['tokens']) if candidates else None
 def audit(self):
  self.qb('QBDIAG');check=self.qb('QBCHECK');trace=Path(self.engine.memory_trace+'.validity.jsonl')
  lines=[json.loads(x) for x in trace.read_text().splitlines() if x]
  if not lines or lines[-1]['missing']!=-1:raise RuntimeError('MTP_ACTIVE_VALIDITY')
  self.journal.emit('state_audit',guard=check['guard'],live_hash=check['live_hash'],mtp_missing=lines[-1]['missing'],native_run=self.engine.memory_run_id)
  return check
 def save(self,ids,n,check):
  # Explicit bounded test scope. Unknown/long opaque native writes are denied.
  # Reserve 4 GiB for <=64K input/checkpoint bundles; oversize results are rejected.
  used=self.usage()
  if len(ids)>self.max_save_tokens or n<self.min_tokens or used+4*1024**3>self.bytes_budget:
   self.journal.emit('checkpoint_skipped',reason='disk_reservation_or_test_scope',disk_bytes=used);return
  ident=uuid.uuid4().hex;d=self.root/'blocks'/ident;d.mkdir(parents=True);start=time.perf_counter();previous=self.entries;published=False
  try:
   self.session('save',d/'state.bin');files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in d.iterdir() if p.is_file()}
   if 'state.bin.g'+str(n) not in files:raise RuntimeError('PINNED_GEOMETRY_MISSING')
   if self.usage()>self.bytes_budget:raise RuntimeError('DISK_BUDGET_EXCEEDED')
   e={'id':ident,'tokens':n,'prefix_ids':ids[:n],'files':files,'guard':check['guard'],'live_hash':check['live_hash'],'created':time.time()};self.entries=[e];self.persist_catalog();published=True
   self.journal.emit('checkpoint_published',checkpoint=ident,prefix_tokens=n,bytes=sum(x['bytes'] for x in files.values()),save_ms=(time.perf_counter()-start)*1000)
   for old in previous:
    olddir=self.root/'blocks'/old['id']
    if len(old['id'])==32 and all(x in '0123456789abcdef' for x in old['id']) and olddir.resolve().parent==(self.root/'blocks').resolve() and olddir!=d and olddir.is_dir():shutil.rmtree(olddir)
  except BaseException as error:
   self.journal.exception('checkpoint_save_failed',error)
   if not published:
    self.entries=previous
    if d.is_dir() and d.resolve().parent==(self.root/'blocks').resolve():shutil.rmtree(d)
 def retention(self):
  # Snapshot bundles have their own signed metadata. These are disposable
  # diagnostics only; never remove an indexed block or the live stderr handle.
  native=self.root/'native'
  if not native.exists():return
  current=getattr(self.engine,'memory_run_id','')
  runs=sorted([p for p in native.iterdir() if p.is_dir()],key=lambda p:p.stat().st_mtime,reverse=True)
  for run in runs[8:]:
   if run.name!=current and len(run.name)==32 and all(c in '0123456789abcdef' for c in run.name) and run.resolve().parent==native.resolve():shutil.rmtree(run)
  for run in runs[:8]:
   files=sorted([p for p in run.iterdir() if p.is_file() and (p.name.endswith('.cells.u64') or '.geometry.r' in p.name)],key=lambda p:p.stat().st_mtime,reverse=True)
   for p in files[30:]:
    if p.resolve().parent==run.resolve():p.unlink()
 def reset_cancelled(self,reason):
  self.warm_images=None;self.warm_text=None
  try:
   if not self.engine.alive():raise RuntimeError('CANCEL_BACKEND_NOT_ALIVE')
   reply=self.qb('QBHOTRESET')
   if not reply.get('weights_retained') or reply.get('live_count')!=0 or not reply.get('guard',{}).get('ok'):raise RuntimeError('CANCEL_RESET_UNVERIFIED')
   status=self.qb('QBSTATUS')
   if status.get('live_count')!=0 or status.get('live_ok'):raise RuntimeError('CANCEL_RESET_UNVERIFIED')
   if getattr(self.engine,'memory_lifecycle',{}).get('cancel_reset_required') is True:raise RuntimeError('CANCEL_RESET_UNVERIFIED')
   self.journal.emit('cancel_reset_verified',reason=reason,weights_retained=True,live_count=0,**snapshot(self.engine))
   return True
  except Exception as error:
   self.journal.exception('cancel_reset_failed',error,reason=reason,**snapshot(self.engine))
   try:self.engine.close()
   except Exception as cleanup:self.journal.exception('backend_cleanup_failed',cleanup)
   self.journal.emit('cancel_backend_discarded',cold_next_request=True)
   return False
 def run(self,ids,thinking,tools,max_new,sampling,cancel,force=None):
  with self.gate:
   start=time.perf_counter();self.retention();rid=getattr(self.journal.local,'request_id',None) or uuid.uuid4().hex;self.journal.local.request_id=rid
   if hasattr(self.gate,'claim_request'):self.gate.claim_request(rid)
   instrument_cancel(cancel,self.journal,rid)
   state=getattr(self.engine,'memory_lifecycle',None)
   if state:state['request_id']=rid
   args=dict(sampling or {});matches=[i for i,x in enumerate(ids) if x==self.im_start];n=matches[-2] if len(matches)>1 else 0
   image=bool(self.image_ids.intersection(ids)) or any(x<0 for x in ids);e=self.choose(ids) if not image else None;restored=False;emitted=False;buffer=[];cache='miss';media=None;warm=False;warm_reason='not_image'
   if image:
    n=matches[-1] if matches else 0
    try:
     emb=getattr(self.svc,'embeddings',None)
     media=media_identity(ids,self.image_pad,getattr(emb,'path',None),getattr(emb,'keys',None))
     warm,warm_reason=warm_image_match(self.warm_images,self.engine.proc,ids,media)
    except Exception as error:self.journal.exception('image_identity_rejected',error);warm_reason='identity_unverified'
    if not self.enabled or self.force_cold:warm=False;warm_reason='memory_disabled'
   else:
    warm,warm_reason=warm_text_match(self.warm_text,self.engine.proc,ids)
    if not self.enabled or self.force_cold:warm=False;warm_reason='memory_disabled'
   proof=self.warm_images if image else self.warm_text
   common=None
   if proof:
    common=0
    for old,new in zip(proof['prefix_ids'],ids):
     if old!=new:break
     common+=1
   self.journal.emit('generation_start',prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(struct.pack('<'+str(len(ids))+'i',*ids)).hexdigest(),max_new=max_new,thinking=bool(thinking),memory_enabled=self.enabled,image_bypass=image and not warm,has_images=image,image_count=len(media or []),resident_candidate=warm,resident_match_reason=warm_reason,resident_common_tokens=common,resident_required_prefix_tokens=len(proof['prefix_ids']) if proof else None,image_identity_sha256=hashlib.sha256(canonical(media)).hexdigest() if media else None)
   try:
    if not self.engine.alive():self.engine.restart()
    state=getattr(self.engine,'memory_lifecycle',None)
    if state:state['request_id']=rid
    self.qb('QBMODE','fast');self.qb('QBGEOMPREFIX','off')
    if warm:
     try:
      if proof['proc'] is not self.engine.proc:raise RuntimeError('RESIDENT_BACKEND_CHANGED')
      current=self.qb('QBCHECK')
      if current['guard']!=proof['check']['guard'] or current['live_hash']!=proof['check']['live_hash']:raise RuntimeError('RESIDENT_STATE_MISMATCH')
      cache='image_warm_prefix' if image else 'text_warm_prefix';self.journal.emit('image_reuse_selected' if image else 'text_reuse_selected',prefix_tokens=len(proof['prefix_ids']),state_verified=True)
     except Exception as error:self.journal.exception('image_reuse_rejected' if image else 'text_reuse_rejected',error);warm=False;warm_reason='native_state_changed'
    self.warm_images=None;self.warm_text=None
    if (e is None or image) and not warm:
     self.qb('QBNEWSEQ');self.journal.emit('native_context_reset',reason=warm_reason)
    if e and self.enabled and not warm:
     try:
      d=self.verify(e);self.session('restore',d/'state.bin');check=self.qb('QBCHECK')
      if check['guard']!=e['guard'] or check['live_hash']!=e['live_hash']:raise RuntimeError('RESTORE_STATE_MISMATCH')
      restored=True;cache='restored';self.journal.emit('restore_verified',checkpoint=e['id'],prefix_tokens=e['tokens'])
     except BaseException as error:
      self.journal.exception('restore_fallback',error,checkpoint=e['id']);self.engine.restart();self.qb('QBNEWSEQ');cache='cold_fallback'
    if self.force_cold or (not self.enabled and e):
     self.qb('QBNEWSEQ')
     if e:
      d=self.verify(e);self.qb('QBGEOMPREFIX',json.dumps(str(d/('state.bin.g'+str(e['tokens']))).replace('\\','/')));n=e['tokens'];cache='full_recompute_recorded_geometry'
    if image:self.journal.emit('checkpoint_skipped',reason='image_persistent_snapshot_disabled',resident_reuse_selected=warm)
    if n>0 and n<len(ids)-1:args['strata_prefix']={'tokens':n}
    # KV-restored answers are held until state/validity audit. Cold ordinary
    # requests retain streaming; a partially emitted request is never replayed.
    heartbeat=time.monotonic()
    with contextlib.closing(self.original_run(ids,thinking,tools,max_new,args,cancel,force=force)) as source:
     for item in source:
      if item[0]=='ping':yield item
      elif restored or warm:
       buffer.append(item)
       if time.monotonic()-heartbeat>=1:heartbeat=time.monotonic();yield ('ping',None)
      else:emitted=True;yield item
    if cancel.is_set():
     self.warm_images=None;self.warm_text=None
     self.journal.emit('generation_cancelled',reason=getattr(cancel,'_kv_reason','caller_cancelled'),audit_skipped=True,already_emitted=emitted,**snapshot(self.engine));self.reset_cancelled(getattr(cancel,'_kv_reason','caller_cancelled'));return
    check=self.audit()
    if image and media and not cancel.is_set() and n>0 and n<len(ids)-1 and not any(x['start']<n<x['end'] for x in media):
     status=self.qb('QBSTATUS')
     if status['live_ok'] and status['live_count']>=n:
      self.warm_images={'proc':self.engine.proc,'prefix_ids':list(ids[:n]),'media':media,'live_count':status['live_count'],'check':check}
      self.journal.emit('image_resident_prefix_verified',prefix_tokens=n,image_count=len(media),live_tokens=status['live_count'])
    if not image and not cancel.is_set() and matches:
     # Match the complete submitted history up to its final assistant header.
     # Keep the original text disk pin n at the previous message boundary;
     # native exact-ID/geometry matching chooses the reusable checkpoint. The
     # proof is not permission to reuse any unverified or divergent suffix.
     r=matches[-1]
     if r>0 and r<len(ids)-1:
      status=self.qb('QBSTATUS')
      if status['live_ok'] and status['live_count']>=r:
       self.warm_text={'proc':self.engine.proc,'prefix_ids':list(ids[:r]),'live_count':status['live_count'],'check':check}
       self.journal.emit('text_resident_prefix_verified',prefix_tokens=r,native_pin_tokens=n,live_tokens=status['live_count'])
    if self.enabled and not image:self.save(ids,n,check)
    for item in buffer:emitted=True;yield item
    last=dict(getattr(self.engine,'last',{}));timing={k:v for k,v in last.items() if isinstance(v,(int,float,str)) and k in ('reused','prompt_read','prompt_tokens','generated','prompt_ms','decode_ms','finish')}
    self.journal.emit('generation_done',cache=cache,restored=restored,image_warm_selected=warm and image,text_warm_selected=warm and not image,image_identity_verified=bool(media),total_ms=(time.perf_counter()-start)*1000,**timing)
   except BaseException as error:
    self.warm_images=None;self.warm_text=None
    if isinstance(error,GeneratorExit) or cancel.is_set():
     if not cancel.is_set():cancel._kv_reason='consumer_closed';cancel.set()
     self.journal.emit('generation_cancelled',reason=getattr(cancel,'_kv_reason','consumer_closed'),audit_skipped=True,consumer_closed=isinstance(error,GeneratorExit),already_emitted=emitted)
     self.reset_cancelled(getattr(cancel,'_kv_reason','consumer_closed'));return
    self.journal.exception('generation_failed',error,cache=cache,already_emitted=emitted,cancelled=cancel.is_set(),cancel_reason=getattr(cancel,'_kv_reason',None),native_error=error_reference(self.engine,error,rid),**snapshot(self.engine,wait=.25))
    try:self.engine.close()
    except Exception as cleanup:self.journal.exception('backend_cleanup_failed',cleanup)
    # Never replay GeneratorExit, an abandoned client, or published model output.
    # GENI embeddings are consumed/unlinked by Service._run: until their lifetime
    # is explicitly owned by the retry, image failures fail closed to a new request.
    retry=isinstance(error,Exception) and (restored or warm) and not emitted and not cancel.is_set() and not image
    if retry:
     self.journal.emit('verified_restore_cold_retry',attempt=1,original_cache=cache)
     try:
      self.engine.restart();self.qb('QBNEWSEQ');args.pop('strata_prefix',None);recovered=[]
      state=getattr(self.engine,'memory_lifecycle',None)
      if state:state['request_id']=rid
      with contextlib.closing(self.original_run(ids,thinking,tools,max_new,args,cancel,force=force)) as retry_source:
       for item in retry_source:
        if item[0]=='ping':yield item
        else:recovered.append(item)
      if cancel.is_set():
       self.journal.emit('cold_retry_cancelled',reason=getattr(cancel,'_kv_reason','caller_cancelled'));self.reset_cancelled(getattr(cancel,'_kv_reason','caller_cancelled'));return
      self.audit()  # A retry is not a permission to publish unchecked state.
      last=dict(getattr(self.engine,'last',{}));timing={k:v for k,v in last.items() if isinstance(v,(int,float,str)) and k in ('reused','prompt_read','prompt_tokens','generated','prompt_ms','decode_ms','finish')}
      self.journal.emit('cold_retry_verified',attempt=1,**timing)
      for item in recovered:emitted=True;yield item
      self.journal.emit('generation_done',cache='verified_cold_retry',restored=False,text_warm_selected=False,image_warm_selected=False,total_ms=(time.perf_counter()-start)*1000,**timing)
     except BaseException as retry_error:
      if isinstance(retry_error,GeneratorExit) or cancel.is_set():
       if not cancel.is_set():cancel._kv_reason='consumer_closed';cancel.set()
       self.journal.emit('cold_retry_cancelled',reason=getattr(cancel,'_kv_reason','consumer_closed'));self.reset_cancelled(getattr(cancel,'_kv_reason','consumer_closed'));return
      self.journal.exception('cold_retry_failed',retry_error,attempt=1,already_emitted=emitted,cancelled=cancel.is_set(),native_error=error_reference(self.engine,retry_error,rid),**snapshot(self.engine,wait=.25))
      try:self.engine.close()
      except Exception as cleanup:self.journal.exception('backend_cleanup_failed',cleanup)
      raise
    else:
     self.journal.emit('cold_retry_suppressed',reason='cancelled' if cancel.is_set() else 'already_published' if emitted else 'image_embedding_lifetime' if image else 'no_reuse_or_generator_closed')
     raise

def install(svc,settings,server):
 c=Controller(svc,settings);base_make=server.make_handler
 def make(s):
  Base=base_make(s)
  class Handler(Base):
   def send_response(self,code,message=None):
    self.kv_response_started=True;self.kv_response_status=code;return super().send_response(code,message)
   def end_headers(self):
    if getattr(self,'kv_request_id',None):self.send_header('X-KV-Request-ID',self.kv_request_id)
    super().end_headers()
   def _json(self,code,body,*args,**kw):
    if code>=400 and isinstance(body,dict):body={**body,'kv_request_id':getattr(self,'kv_request_id',None)}
    return super()._json(code,body,*args,**kw)
   def _watch_client(self,cancel):
    self.kv_cancel=instrument_cancel(cancel,c.journal,self.kv_request_id)
    return super()._watch_client(cancel)
   def _note(self,**values):
    if values.get('outcome')=='disconnected':
     self.kv_outcome='cancelled'
     if hasattr(self,'kv_cancel'):self.kv_cancel._kv_reason='client_disconnected'
     c.journal.emit('client_disconnected',request_id=self.kv_request_id,stream_started=getattr(self,'kv_stream',False))
    if values.get('error'):
     self.kv_outcome='failed';c.journal.emit('sse_failed' if getattr(self,'kv_stream',False) else 'response_failed',request_id=self.kv_request_id,http_status=getattr(self,'kv_response_status',None),error_code='MODEL_OR_STREAM_ERROR')
    parent=getattr(super(),'_note',None)
    if parent:return parent(**values)
   def _sse(self,*a,**kw):
    self.kv_stream=True
    return super()._sse(*a,**kw)
   def do_POST(self):
    self.kv_outcome='completed';self.kv_stream=False
    self.kv_request_id=uuid.uuid4().hex;c.journal.local.request_id=self.kv_request_id;start=time.perf_counter();c.journal.emit('http_start',path=self.path.split('?')[0],content_length=self.headers.get('Content-Length'))
    try:
     with c.gate:super().do_POST()
    except BaseException as e:
     if isinstance(e,ConnectionError):
      self.kv_outcome='cancelled'
      cancel=getattr(self,'kv_cancel',None)
      if cancel is not None:cancel._kv_reason='client_disconnected';cancel.set()
      c.journal.emit('http_cancelled',reason='client_disconnected',error_type=type(e).__name__);return
     self.kv_outcome='failed';c.journal.exception('http_failed',e,stream_started=self.kv_stream)
     if not getattr(self,'kv_response_started',False):self._json(503,{'error':{'type':'server_error','message':'KV request failed; report the request ID.','request_id':self.kv_request_id}})
     elif self.kv_stream:
      c.journal.emit('sse_failed',http_status=getattr(self,'kv_response_status',None),error_code='KV_REQUEST_FAILED')
      error={'type':'server_error','code':'KV_REQUEST_FAILED','message':'Request failed safely; retry with the complete history.','request_id':self.kv_request_id}
      if self.path.split('?')[0].endswith('/responses'):
       frame='event: response.failed\ndata: '+json.dumps({'type':'response.failed','response':{'status':'failed','error':error}})+'\n\n'
      elif self.path.split('?')[0].endswith('/messages'):
       frame='event: error\ndata: '+json.dumps({'type':'error','error':error})+'\n\n'
      else:frame='data: '+json.dumps({'error':error})+'\n\ndata: [DONE]\n\n'
      try:self.wfile.write(frame.encode());self.wfile.flush();self.close_connection=True
      except OSError as write_error:c.journal.emit('sse_write_failed',error_type=type(write_error).__name__)
     else:raise
    finally:
     cancel=getattr(self,'kv_cancel',None)
     if cancel is not None and cancel.is_set() and self.kv_outcome=='completed':self.kv_outcome='cancelled'
     c.journal.emit('http_end',status=getattr(self,'kv_response_status',None),outcome=self.kv_outcome,stream=self.kv_stream,wall_ms=(time.perf_counter()-start)*1000)
  return Handler
 server.make_handler=make
 class Control(BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_GET(self):
   body={'status':'ok','test_environment':True,'memory_enabled':c.enabled,'transaction_active':bool(c.gate.ident),'engine_interface':1,'blocks':len(c.entries),'disk_budget_bytes':c.bytes_budget,'log_directory':str(c.root/'logs'),'recent_request_ids':list(dict.fromkeys(x['request_id'] for x in c.journal.last if x.get('request_id')))[-10:]};raw=canonical(body);self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
  def do_POST(self):
   if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+svc.api_key):self.send_response(401);self.end_headers();return
   with c.gate:
    if self.path=='/memory/enable':c.enabled=True;c.force_cold=False
    elif self.path=='/memory/disable':c.enabled=False;c.force_cold=True
    else:self.send_response(404);self.end_headers();return
    c.journal.emit('memory_mode_changed',enabled=c.enabled);self.do_GET()
 httpd=ThreadingHTTPServer(('127.0.0.1',int(settings['port'])),Control);c.httpd=httpd;threading.Thread(target=httpd.serve_forever,daemon=True).start();c.journal.emit('gateway_ready',port=settings['port'],namespace=c.namespace,test_environment=True)
 return c
