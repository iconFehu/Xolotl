"""Native ERR allowlist: store templates/numbers, never arbitrary native text.

Unknown payloads (including URLs, paths, credentials and prompt fragments) are
omitted. A deployment-local keyed fingerprint correlates repeated errors without
publishing a dictionary-testable hash of potentially sensitive error text.
"""
import hashlib, hmac, re

_FIXED = {
 'CANCEL_RESET_REQUIRED': ('CANCEL_RESET_REQUIRED','cancellation'),
 'QSA prototype session is not healthy': ('QSA_SESSION_UNHEALTHY','state'),
 'isolated resume requires empty released archive bank, checkpoints and real MTP ring': ('QSA_RESUME_PRECONDITION','state'),
 'selective base requires exact live-prefix extension': ('QSA_SELECTIVE_PREFIX','identity'),
 'QSA prototype requires exact append-only committed raw IDs and ckpt=1 pin=0': ('QSA_APPEND_ONLY_REQUIRED','identity'),
 'QSA prototype refuses changed image identities in committed prefix': ('QSA_IMAGE_IDENTITY','identity'),
 'QSA prototype refuses a checkpoint mount or changed control-vector state': ('QSA_CHECKPOINT_OR_CONTROL_IDENTITY','identity'),
 'QSA prototype cannot fall back to prefill or a checkpoint': ('QSA_PREFILL_REFUSED','state'),
 'selective checkpoint crossed external range': ('QSA_EXTERNAL_RANGE','state'),
 'restoring a conversation checkpoint failed': ('CHECKPOINT_RESTORE_FAILED','checkpoint'),
 'the K/V cannot grow to this prompt: no VRAM is left': ('KV_CAPACITY_OOM','capacity'),
 'the K/V cannot grow to the parked conversation: no VRAM is left': ('KV_CAPACITY_OOM','capacity'),
 'the K/V cannot grow: no VRAM is left': ('KV_CAPACITY_OOM','capacity'),
 'a token id is outside the vocabulary': ('TOKEN_OUT_OF_VOCABULARY','request'),
 'bad request: max_new': ('INVALID_MAX_NEW','request'),
 'an adaptive refill failed': ('EXPERT_REFILL_FAILED','expert_cache'),
 'fixed diagnostic policy needs real MTP capacity4': ('MTP_CAPACITY_REQUIRED','mtp'),
 'diagnostic commit fence failed': ('DIAGNOSTIC_COMMIT_FAILED','synchronization'),
 'stage sync': ('CUDA_STAGE_SYNC_FAILED','synchronization'),
 'cancelled': ('NATIVE_CANCELLED','cancellation'),
}
_GEOMETRY_DETAILS = {'token count','span count','noncontiguous span','incomplete span coverage',
 'unit/part disagreement','MTP write metadata','image positions','prefix length',
 'input header','input token','input span','input MTP unit','actual MTP unit absent',
 'prefix MTP unit absent','prefix MTP capacity/window identity','actual unit/path replay differs',
 'actual prefix replay differs','output open','output write','input open'}

def message_body(line):
    text=str(line).strip()
    if text.startswith('ERR '):return text[4:].strip()
    if text.startswith('FATAL '):return text[6:].strip()
    return text

def fingerprint(text,key):
    if not isinstance(key,bytes) or len(key)!=32:raise ValueError('NATIVE_LOG_KEY_INVALID')
    return hmac.new(key,message_body(text).encode('utf-8','replace'),'sha256').hexdigest()

def classify_native_error(line,key):
    body=message_body(line);params={};detail=None;code='NATIVE_ERR_UNCLASSIFIED';family='unknown'
    if body in _FIXED:code,family=_FIXED[body]
    else:
        match=re.fullmatch(r'GEOMETRY_LINEAGE_MISS resume=([0-9]{1,7})( identity)?',body)
        mtp=re.fullmatch(r'MTP_VALIDITY_MISS resume=(-?[0-9]{1,7}) q=(-?[0-9]{1,7}) missing=(-?[0-9]{1,7})',body)
        if match and int(match[1])<=524288:
            code='GEOMETRY_LINEAGE_IDENTITY' if match[2] else 'GEOMETRY_LINEAGE_MISSING';family='geometry';params={'resume':int(match[1])}
        elif mtp and all(-32832<=int(x)<=524288 for x in mtp.groups()):
            code='MTP_VALIDITY_MISS';family='mtp';params=dict(zip(('resume','q','missing'),map(int,mtp.groups())))
        elif body.startswith('GEOMETRY_MISMATCH: ') and body[19:] in _GEOMETRY_DETAILS:
            code='GEOMETRY_MISMATCH';family='geometry';detail=body[19:]
        elif re.fullmatch(r'verify(?: batch)?: timed out at layer [0-9]{1,4}',body):
            code='VERIFY_TIMEOUT';family='verifier';params={'layer':int(body.rsplit(' ',1)[1])}
        elif body in ('verify: an earlier window never finished','verify batch: an earlier window never finished'):
            code='VERIFY_PREVIOUS_WINDOW_UNFINISHED';family='verifier'
        else:
            context=re.fullmatch(r'prompt \(([0-9]{1,7}) tokens\) \+ max_new \(([0-9]{1,7})\) exceeds the context \(([0-9]{1,7})\)',body)
            if context and all(int(x)<=1048576 for x in context.groups()):
                code='CONTEXT_LIMIT';family='request';params=dict(zip(('prompt_tokens','max_new','context'),map(int,context.groups())))
    summary=code
    if params:summary+=' '+' '.join(f'{k}={v}' for k,v in params.items())
    if detail:summary+=' '+detail
    if family=='unknown':summary+=' [native message omitted]'
    return {'code':code,'family':family,'parameters':params,'detail':detail,'summary':summary,
            'message_hmac_sha256':fingerprint(body,key),'message_bytes':len(body.encode('utf-8','replace')),
            'raw_message_omitted':True}

def error_reference(engine,error,request_id):
    state=getattr(engine,'memory_lifecycle',None) or {};last=state.get('native_error')
    # The native pipe can produce errors outside a generation. Only associate a
    # ValueError with the same request and the exact sanitized frame it consumed.
    if not isinstance(error,ValueError) or not last or last.get('request_id')!=request_id:return None
    if str(error).strip()!=last['summary']:return None
    return {k:v for k,v in last.items() if k not in ('summary',)}
