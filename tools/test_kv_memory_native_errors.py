"""Classifier, privacy and real native-stdout protocol regression (model-free)."""
import json,sys,tempfile,subprocess,queue,threading,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serve/kv_memory'))
from kv_live import Journal
from observability import engine_class
from native_errors import classify_native_error,error_reference
key=os.urandom(32);checks=[]
cases=[('ERR GEOMETRY_LINEAGE_MISS resume=221082 identity','GEOMETRY_LINEAGE_IDENTITY'),
 ('ERR GEOMETRY_LINEAGE_MISS resume=32768','GEOMETRY_LINEAGE_MISSING'),
 ('ERR MTP_VALIDITY_MISS resume=16384 q=20000 missing=19000','MTP_VALIDITY_MISS'),
 ('ERR GEOMETRY_MISMATCH: prefix length','GEOMETRY_MISMATCH'),
 ('ERR bad request: max_new','INVALID_MAX_NEW'),
 ('ERR the K/V cannot grow: no VRAM is left','KV_CAPACITY_OOM'),
 ('ERR prompt (520000 tokens) + max_new (8192) exceeds the context (524288)','CONTEXT_LIMIT'),
 ('ERR verify: timed out at layer 15','VERIFY_TIMEOUT'),
 ('ERR verify batch: an earlier window never finished','VERIFY_PREVIOUS_WINDOW_UNFINISHED'),
 ('ERR QSA prototype refuses changed image identities in committed prefix','QSA_IMAGE_IDENTITY')]
for raw,expected in cases:
 out=classify_native_error(raw,key);assert out['code']==expected,(raw,out)
assert classify_native_error(cases[0][0],key)['parameters']=={'resume':221082}
checks.append('ten_allowlisted_families_and_safe_numbers')
secrets=['KEY_SENTINEL_8d436f','Bearer private-token','C:\\Users\\PrivateUser\\secret.txt',
 'https://private.example/path?api_key=secret','prompt: PRIVATE_HISTORY','password=private-secret']
unknown='ERR unrecognized backend error '+' | '.join(secrets)
for raw in (unknown,'ERR GEOMETRY_LINEAGE_MISS resume=221082 identity '+unknown,
            unknown*1000,'ERR GEOMETRY_LINEAGE_MISS resume=9999999 identity','ERR '+unknown+'\nERR bad request: max_new'):
 out=classify_native_error(raw,key);encoded=json.dumps(out)
 assert out['code']=='NATIVE_ERR_UNCLASSIFIED' and all(secret not in encoded for secret in secrets)
 assert len(encoded)<700 and out['raw_message_omitted']
a=classify_native_error(unknown,key);assert a['message_hmac_sha256']==classify_native_error(unknown,key)['message_hmac_sha256']
assert a['message_hmac_sha256']!=classify_native_error(unknown,os.urandom(32))['message_hmac_sha256']
checks.append('unknown_multiline_oversize_suffix_credentials_paths_urls_omitted_keyed_fingerprint')
with tempfile.TemporaryDirectory() as temp:
 journal=Journal(Path(temp)/'logs')
 class Base:
  def close(self):
   if self.proc.poll() is None:self.proc.terminate();self.proc.wait(timeout=5)
 E=engine_class(Base,journal,())
 for index,raw in enumerate((cases[0][0],unknown)):
  e=E();e.memory_run_id=str(index);e.lines=queue.Queue();e.slot_q=[];e.ended=False
  code='import time;time.sleep(.2);print('+repr(raw)+',flush=True)'
  e.proc=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
  t=threading.Thread(target=e._pump);t.start()
  for _ in range(100):
   if hasattr(e,'memory_lifecycle'):break
   time.sleep(.005)
  e.memory_lifecycle.update(request_id='request-'+str(index),operation='GEN')
  line=e.lines.get(timeout=5);t.join(5);assert not t.is_alive()
  assert line.startswith('ERR ') and all(secret not in line for secret in secrets)
  ref=error_reference(e,ValueError(line[4:].strip()),'request-'+str(index));assert ref and ref['operation']=='GEN'
  assert error_reference(e,ValueError(line[4:].strip()),'other-request') is None
  assert error_reference(e,ValueError('unrelated error'),'request-'+str(index)) is None
  journal.exception('generation_failed',ValueError(line[4:].strip()),native_error=ref)
  e.proc.wait(timeout=5);e.proc.stdout.close();e.proc.stdin.close()
 rows=[json.loads(s) for s in (Path(temp)/'logs/events.jsonl').read_text().splitlines()]
 err=[x for x in rows if x['event']=='native_error'];assert len(err)==2
 assert err[0]['code']=='GEOMETRY_LINEAGE_IDENTITY' and err[0]['parameters']['resume']==221082
 assert len({x['error_id'] for x in err})==2
 failures=[x for x in rows if x['event']=='generation_failed'];assert failures[0]['error_code']=='GEOMETRY_LINEAGE_IDENTITY'
 text=(Path(temp)/'logs/events.jsonl').read_text();assert all(secret not in text for secret in secrets)
 assert journal.native_error_key==Journal(Path(temp)/'logs').native_error_key
checks+=['real_child_stdout_is_sanitized_before_consumer','request_operation_event_id_and_ValueError_association','stale_error_not_attributed','native_log_key_persists_privately']
print(json.dumps({'passed':True,'checks':checks,'GPU_calls':0}))
