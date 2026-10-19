import asyncio, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
from vagen.envs.active_spatial.visual_bbox_metrics import compute_visual_bbox_metrics, _camera_pose_from_forward

repo = Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
out = repo/'exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_stage2_renderer_validation_20260902/r1_overlays'
out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in open(repo/'exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl') if x.strip()]
types=['projective_relations','fov_inclusion','occlusion_alignment']
selected=[]
for t in types:
    selected += [r for r in rows if r['scene_id']=='0226_840298' and r['task_type']==t][:10]
states=[]
for i,r in enumerate(selected):
    states.append((i,r,'init',np.asarray(r['init_camera']['extrinsics'],np.float32)))
    states.append((i,r,'target',_camera_pose_from_forward(np.asarray(r['sample_target']),np.asarray(r['camera_params']['forward'])).astype(np.float32)))
def kh1(k):
    x=np.asarray(k,np.float32).copy(); x[0,:]*=.4; x[1,:]*=256/480; return x
tasks=[]
for _,r,_,pose in states:
    k=np.asarray(r['init_camera']['intrinsics'],np.float32); w2c=np.linalg.inv(pose).astype(np.float32)
    tasks += [{'mode':'cam_param','intrinsics':k.tolist(),'extrinsics':w2c.tolist(),'size':[256,256]}, {'mode':'cam_param','intrinsics':kh1(k).tolist(),'extrinsics':w2c.tolist(),'size':[256,256]}]
renderer=UnifiedRenderGS(render_backend='http',client_url='http://127.0.0.1:8876',scene_id='0226_840298')
images=asyncio.run(renderer.render_tasks(tasks))
manifest=[]; thumbs=[]
for j,(i,r,phase,pose) in enumerate(states):
    for name,k,img in [('H0',np.asarray(r['init_camera']['intrinsics'],np.float32),images[2*j]),('H1',kh1(r['init_camera']['intrinsics']),images[2*j+1])]:
        params=dict(r.get('task_params') or {}); params.update({'_target_object':r['target_object'],'_camera_intrinsics':k.tolist(),'_image_width':256,'_image_height':256,'_camera_pose_c2w':pose.tolist()})
        vm=compute_visual_bbox_metrics(pose[:3,3],pose[:3,2],r['task_type'],params,r['target_region'])
        im=img.copy(); d=ImageDraw.Draw(im)
        for o in vm.get('objects',[]) or []:
            b=o.get('bbox')
            if b and b[2] >= b[0] and b[3] >= b[1]:
                d.rectangle(b,outline=(255,0,0),width=2); d.text((b[0],max(0,b[1]-12)),str(o.get('label',o.get('id','obj'))),fill=(255,255,0))
        d.text((3,3),f'{i} {r["task_type"]} {phase} {name}',fill=(255,255,0))
        fn=f'{j:03d}_{i:02d}_{phase}_{name}.png'; im.save(out/fn)
        manifest.append({'sample_index':i,'state_index':j,'phase':phase,'camera_model':name,'task_type':r['task_type'],'scene_id':r['scene_id'],'image':fn,'visual_metrics':vm,'pose_c2w':pose.tolist(),'K':k.tolist()})
        thumbs.append(im.resize((128,128)))
sheet=Image.new('RGB',(1280,((len(thumbs)+9)//10)*128),'white')
for i,im in enumerate(thumbs): sheet.paste(im,((i%10)*128,(i//10)*128))
sheet.save(out/'contact_sheet.png')
with open(out/'r1_overlay_manifest.jsonl','w') as f:
    for x in manifest:f.write(json.dumps(x)+'\n')
print(json.dumps({'states':len(states),'renders':len(manifest),'tasks':{t:sum(x['task_type']==t for x in manifest) for t in types}}))
