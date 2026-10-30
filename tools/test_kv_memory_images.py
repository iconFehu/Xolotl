import sys,json,struct,tempfile,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serve/kv_memory'))
from kv_live import media_identity,warm_image_match
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'image.bin';header=struct.pack('<5i',0x31455653,4,2,2,2);payload=struct.pack('<8f',*range(8));p.write_bytes(header+payload)
    ids=[1,10,3,2,2,2,2,11,1,12,13,14];key=['1234567890abcdef']
    media=media_identity(ids,2,p,key);proc=object();warm={'proc':proc,'prefix_ids':ids[:8],'media':media,'live_count':12}
    assert warm_image_match(warm,proc,ids+[20,21],media)[0]
    # Identical tokenized image placeholders are insufficient: actual pixels /
    # encoder bytes, grid, order, positions, history and native owner all bind.
    p.write_bytes(header+payload[:-1]+bytes([payload[-1]^1]));changed=media_identity(ids,2,p,key)
    assert not warm_image_match(warm,proc,ids+[20,21],changed)[0]
    assert not warm_image_match(warm,object(),ids+[20,21],media)[0]
    altered=list(ids);altered[1]=99;assert not warm_image_match(warm,proc,altered+[20,21],media)[0]
    for field,value in [('start',4),('grid',[4,1,4,2]),('key','f'*16)]:
        altered=copy.deepcopy(media);altered[0][field]=value;assert not warm_image_match(warm,proc,ids+[20,21],altered)[0]
    assert not warm_image_match(warm,proc,ids+[20,21],[])[0]
    # A new image appearing before the old live extent must cold-reset because
    # the native committed-prefix image guard also covers that extent.
    assert not warm_image_match(warm,proc,ids+[20,21],media+[{**media[0],'start':10,'end':14}])[0]
    assert warm_image_match(warm,proc,ids+[20,21],media+[{**media[0],'start':13,'end':17}])[0]
    rejected=0
    for content,keys in [(header+payload[:-1],key),(header+payload+b'x',key),(header+payload,[]),(header+payload,['invalid']),(struct.pack('<5i',0x31455653,4,1,2,2)+payload,key)]:
        p.write_bytes(content)
        try:media_identity(ids,2,p,keys)
        except RuntimeError:rejected+=1
    assert rejected==5
    p.write_bytes(header+payload+header+struct.pack('<8f',*range(1,9)))
    multi=[1,3,2,2,2,2,4,3,2,2,2,2,4,1,12,13]
    two=media_identity(multi,2,p,key+['fedcba0987654321']);assert len(two)==2
    multiwarm={'proc':proc,'prefix_ids':multi[:13],'media':two,'live_count':16}
    assert warm_image_match(multiwarm,proc,multi+[20,21],two)[0]
    assert not warm_image_match(multiwarm,proc,multi+[20,21],list(reversed(two)))[0]
    print(json.dumps({'passed':True,'identity_scenarios':17,'malformed_media_refused':rejected,'multiple_images':True,'GPU_calls':0}))
