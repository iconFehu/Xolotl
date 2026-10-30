"""The disabled server hook must not import/configure the experimental adapter."""
import ast
from pathlib import Path
import types

source=Path(__file__).resolve().parents[1]/'serve/server.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
initialize=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_memory_adapter' for t in n.targets))
configure=next(n for n in main.body if isinstance(n,ast.If) and 'memory_gateway' in ast.unparse(n.test))
install=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='_memory_adapter is not None')
for cfg in ({},{'memory_gateway':None},{'memory_gateway':{}},{'memory_gateway':{'enabled':False}}):
    server=types.SimpleNamespace(StrataEngine=object())
    namespace={'cfg':cfg,'sys':types.SimpleNamespace(modules={'test_server':server}),'__name__':'test_server'}
    original=server.StrataEngine
    exec(compile(ast.fix_missing_locations(ast.Module(body=[initialize,configure,install],type_ignores=[])),str(source),'exec'),namespace)
    assert namespace['_memory_adapter'] is None and server.StrataEngine is original
    assert '_memory_import' not in namespace and 'svc' not in namespace
print('KV_MEMORY_DEFAULT_OFF_PASS: no adapter import, engine replacement or service installation')
