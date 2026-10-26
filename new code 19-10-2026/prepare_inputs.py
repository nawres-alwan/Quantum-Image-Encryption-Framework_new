"""Copy the four original source images losslessly, recording exact provenance."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

SOURCE=Path(__file__).resolve().parent.parent / 'images encryption and decryption models'
FILES=[('IM1','Ashurbanipal','code/Ashurbanipal.jpg'),('IM2','Stele of Hammurabi','IM2_original.png'),
       ('IM3','Lamassu','IM3_original.png'),('IM4','Lion of Babylon','IM4_original.jpg')]


def prepare(root):
    root=Path(root);inputs=root/'inputs';inputs.mkdir(exist_ok=True)
    saved=root/'input_manifest.json'
    if saved.exists():
        rows=json.loads(saved.read_text(encoding='utf-8'))
        for rec in rows:
            local=root/rec['relative_input']
            if hashlib.sha256(local.read_bytes()).hexdigest()!=rec['file_sha256']:
                raise ValueError(f'Packaged input changed: {local}')
            original=Path(rec['original_source'])
            if not original.is_absolute(): original=SOURCE.parent/original
            if original.exists() and hashlib.sha256(original.read_bytes()).hexdigest()!=rec['file_sha256']:
                raise ValueError(f'Original input changed: {original}')
        return rows
    rows=[]
    for image_id,title,relative in FILES:
        source=SOURCE/relative;dest=inputs/(image_id+'_'+source.name)
        if dest.exists() and dest.read_bytes()!=source.read_bytes():
            raise FileExistsError(f'Existing input differs: {dest}')
        if not dest.exists(): shutil.copy2(source,dest)
        with Image.open(dest) as im: array=np.array(im.convert('RGB'))
        rows.append({'image_id':image_id,'title':title,'relative_input':str(dest.relative_to(root)),
                     'original_source':str(source),'file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'rgb_pixel_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                     'height':array.shape[0],'width':array.shape[1],'channels':3})
    (root/'input_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows


if __name__=='__main__':
    print(json.dumps(prepare(Path(__file__).resolve().parent),indent=2))
