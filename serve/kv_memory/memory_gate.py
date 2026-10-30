"""One shared service/native transaction gate, including ordinary/direct callers.

Lock order: MemoryGate -> Service FIFO -> native stream. A transaction cannot be
held across unrelated HTTP requests. Restore/generate/save run in one scope.
"""
import contextlib,json,threading,time,uuid

class FencedInput:
 def __init__(self,gate,raw):self.gate=gate;self.raw=raw
 def write(self,data):
  kind=data.split(' ',1)[0].split('\n',1)[0]
  if kind in {'GEN','GENI','SAVE','RESTORE'} or kind.startswith('QB'):
   self.gate.require_owner();data='MEM_CALL '+self.gate.ident+' '+data
  return self.raw.write(data)
 def __getattr__(self,name):return getattr(self.raw,name)

class GuardedQueue:
 def __init__(self,gate,raw):self.gate=gate;self.raw=raw
 def get(self,*a,**kw):
  line=self.raw.get(*a,**kw)
  if isinstance(line,str) and line.startswith('MEMRESULT ') and not getattr(self.gate.local,'reading_control',False):
   engine=self.gate.owner.engine
   if hasattr(engine,'_silent'):raise engine._silent('MEMORY_UNEXPECTED_NATIVE_RESPONSE')
   raise RuntimeError('MEMORY_UNEXPECTED_NATIVE_RESPONSE')
  return line
 def __getattr__(self,name):return getattr(self.raw,name)

class MemoryGate:
 def __init__(self,owner):
  self.owner=owner;self.lock=threading.RLock();self.local=threading.local();self.proc=None;self.raw=None;self.ident=None;self.events=[];self.sequence=0
 def exchange(self,line):
  self.raw.write(line+'\n');self.raw.flush();deadline=time.monotonic()+120
  while True:
   self.local.reading_control=True
   try:item=self.owner.engine.lines.get(timeout=max(.01,deadline-time.monotonic()))
   finally:self.local.reading_control=False
   if item is None:raise RuntimeError('MEMORY_BACKEND_EOF')
   if not item.startswith('MEMRESULT '):raise RuntimeError('MEMORY_PROTOCOL_DESYNC')
   answer=json.loads(item[10:])
   if answer.get('version')!=1 or not answer.get('ok'):raise RuntimeError('MEMORY_PROTOCOL_REJECTED:'+str(answer.get('error')))
   return answer
 def require_owner(self):
  if not getattr(self.local,'depth',0) or not self.ident or self.proc is not self.owner.engine.proc:raise RuntimeError('MEMORY_TRANSACTION_REQUIRED')
 def bind(self):
  p=self.owner.engine.proc
  if p is self.proc and self.ident:return
  if self.ident:self.events.append({'event':'backend_discarded','sequence':self.sequence})
  self.ident=None
  self.proc=p;self.raw=p.stdin.raw if isinstance(p.stdin,FencedInput) else p.stdin
  if not isinstance(self.owner.engine.lines,GuardedQueue):self.owner.engine.lines=GuardedQueue(self,self.owner.engine.lines)
  p.stdin=FencedInput(self,self.raw);caps=self.exchange('MEM_CAPS')
  if not all(caps.get(k) is True for k in ('geometry_replay','mtp_active_audit','transaction_fence')):raise RuntimeError('MEMORY_CAPABILITY_MISSING')
  ident=uuid.uuid4().hex;reply=self.exchange('MEM_BEGIN '+ident);self.ident=ident;self.sequence+=1
  self.events.append({'event':'begin','sequence':self.sequence,'pid':p.pid,'native_epoch':reply['epoch'],'thread':threading.get_ident()})
 def __enter__(self):
  self.lock.acquire();self.local.depth=getattr(self.local,'depth',0)+1
  try:
   p=self.owner.engine.proc
   if p is not None and p.poll() is None:self.bind()
   return self
  except BaseException:
   self.ident=None
   try:self.owner.engine.close()
   finally:self.local.depth-=1;self.lock.release()
   raise
 def __exit__(self,*error):
  try:
   if self.local.depth==1 and self.ident:
    if self.proc is self.owner.engine.proc and self.proc.poll() is None:
     reply=self.exchange('MEM_END '+self.ident);self.events.append({'event':'end','sequence':self.sequence,'pid':self.proc.pid,'native_epoch':reply['epoch'],'thread':threading.get_ident()})
    else:self.events.append({'event':'backend_discarded','sequence':self.sequence})
    self.ident=None
  except BaseException:
   self.ident=None;self.owner.engine.close();raise
  finally:self.local.depth-=1;self.lock.release()
 def install_service(self,service):
  # Wrap BEFORE Service acquires its own FIFO, avoiding FIFO->gate inversion.
  for name in ('run',):
   original=getattr(service,name)
   def generate(*args,_original=original,**kwargs):
    with self:yield from _original(*args,**kwargs)
   setattr(service,name,generate)
  for name in ('slot_action','unload_waiting','unload','load'):
   if not hasattr(service,name):continue
   original=getattr(service,name)
   def call(*args,_original=original,**kwargs):
    with self:return _original(*args,**kwargs)
   setattr(service,name,call)
 @contextlib.contextmanager
 def transaction(self):
  with self:yield self
