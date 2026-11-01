"""Privacy preserving process, cancellation and transport outcome records.

Only classifiers/hashes enter the journal. Native stdout can contain tokens:
never copy arbitrary stdout, exception strings, argv or request bodies here.
"""
import hashlib, threading, time, uuid
from native_errors import classify_native_error

def snapshot(engine, wait=0):
    state = getattr(engine, 'memory_lifecycle', None)
    proc = getattr(engine, 'proc', None)
    if state is None:
        return {'engine_pid': getattr(proc, 'pid', None), 'native_run': getattr(engine, 'memory_run_id', None),
                'exit_code': proc.poll() if proc else None}
    if wait and proc is state['proc']:
        try: proc.wait(timeout=wait)
        except Exception: pass
    rc = state['proc'].poll()
    return {'engine_pid': state['proc'].pid, 'native_run': state['run'], 'exit_code': rc,
            'exit_code_hex': None if rc is None else f'0x{rc & 0xffffffff:08X}',
            'termination_cause': state['cause'], 'stdout_eof': state['eof'],
            'fatal_code': state.get('fatal'), 'fatal_sha256': state.get('fatal_sha256')}

def instrument_cancel(cancel, journal, rid):
    if getattr(cancel, '_kv_observed', False): return cancel
    cancel._kv_observed = True
    original = cancel.set
    def set_cancel():
        if not cancel.is_set():
            cancel._kv_reason = getattr(cancel, '_kv_reason', 'caller_cancelled')
            journal.emit('request_cancelled', request_id=rid, reason=cancel._kv_reason)
        original()
    cancel.set = set_cancel
    return cancel

def engine_class(Base, journal, fatal_prefixes):
    class ObservedEngine(Base):
        def _pump(self):
            proc, lines, slot_q = self.proc, self.lines, self.slot_q
            state = self.memory_lifecycle = {'proc': proc, 'run': self.memory_run_id,
                                            'cause': 'native_exit_or_external_termination', 'eof': False}
            def reap():
                rc = proc.wait()
                journal.emit('native_exit', request_id=state.get('request_id'), engine_pid=proc.pid,
                             native_run=state['run'], exit_code=rc, exit_code_hex=f'0x{rc & 0xffffffff:08X}',
                             termination_cause=state['cause'], stdout_eof=state['eof'],
                             fatal_code=state.get('fatal'), fatal_sha256=state.get('fatal_sha256'))
            threading.Thread(target=reap, daemon=True, name='kv-native-exit').start()
            line = None
            try:
                for line in proc.stdout:
                    safe_error = None
                    if line.startswith(('ERR', 'FATAL ')):
                        safe_error = classify_native_error(line,journal.native_error_key)
                        safe_error.update(error_id=uuid.uuid4().hex,request_id=state.get('request_id'),
                                          operation=state.get('operation','unknown'),native_run=state['run'],engine_pid=proc.pid)
                        state['native_error'] = safe_error
                        journal.emit('native_error',**safe_error)
                    if line.startswith(fatal_prefixes):
                        state.update(cause='native_verifier_fatal', fatal='VERIFY_TIMEOUT' if 'timed out' in line else 'VERIFY_PREVIOUS_WINDOW_UNFINISHED',
                                     fatal_sha256=hashlib.sha256(line.encode()).hexdigest())
                        journal.emit('native_fatal', request_id=state.get('request_id'), **snapshot(self))
                        if self.proc is proc:
                            self.silent_note = 'KV_NATIVE_VERIFIER_FATAL'; self.ended = True
                        try: proc.terminate()
                        except OSError: pass
                        line='ERR '+safe_error['summary']+'\n'
                        break
                    if line.startswith(('BT ', 'BDONE ')) and slot_q:
                        try: slot_q[int(line.split()[1])].put(line); continue
                        except (IndexError, ValueError): pass
                    # Downstream ValueError/public SSE/server stdout must never
                    # re-emit an arbitrary native payload. Preserve ERR framing.
                    if safe_error:line=('FATAL ' if line.startswith('FATAL ') else 'ERR ')+safe_error['summary']+'\n'
                    lines.put(line)
            except (OSError, ValueError) as error:
                journal.emit('native_stdout_failed', request_id=state.get('request_id'), error_type=type(error).__name__,
                             engine_pid=proc.pid, native_run=state['run'])
            finally:
                state['eof'] = True
                if self.proc is proc:
                    self.ended = True
                    if line and line.startswith('ERR'): self.last_err = line[4:].strip()
                journal.emit('native_stdout_eof', request_id=state.get('request_id'), **snapshot(self))
                lines.put(None)
                for q in slot_q: q.put(None)
        def _silent(self, what):
            state = getattr(self, 'memory_lifecycle', None)
            if state: state['cause'] = 'server_silence_or_protocol_guard'
            journal.emit('native_guard_termination', **snapshot(self), reason_sha256=hashlib.sha256(what.encode()).hexdigest())
            return super()._silent(what)
        def close(self):
            state = getattr(self, 'memory_lifecycle', None)
            if state and self.proc is state['proc'] and self.proc.poll() is None:
                # An earlier EOF/fatal remains the primary cause, never relabel it as a normal close.
                if not state['eof'] and state['cause'] == 'native_exit_or_external_termination':
                    state['cause'] = 'server_close_requested'
                journal.emit('native_close_requested', **snapshot(self))
            try: return super().close()
            finally:
                if state:
                    rc=state['proc'].poll()
                    journal.emit('native_close_completed', engine_pid=state['proc'].pid, native_run=state['run'],
                                 exit_code=rc, exit_code_hex=None if rc is None else f'0x{rc & 0xffffffff:08X}',
                                 termination_cause=state['cause'])
    return ObservedEngine
