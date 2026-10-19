import asyncio,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
from vagen.envs.active_spatial.visual_bbox_metrics import compute_visual_bbox_metrics,_camera_pose_from_forward
repo=Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite'); out=repo/'exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_stage2_renderer_validation_20260902/projective_real_100'; out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in open(repo/'exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl') if x.strip()]; rows=[r for r in rows if r['scene_id']=='0226_840298' and r['task_type']=='projective_relations'][:100]
def eff(k):
 k=np.asarray(k,np.float32).copy();k[0,:]*=.4;k[1,:]*=256/480;return k
tasks=[]; poses=[]
for r in rows:
 p=_camera_pose_from_forward(np.asarray(r['sample_target']),np.asarray(r['camera_params']['forward'])).astype(np.float32);poses.append(p);tasks.append({'mode':'cam_param','intrinsics':eff(r['init_camera']['intrinsics']).tolist(),'extrinsics':np.linalg.inv(p).astype(np.float32).tolist(),'size':[256,256]})
rr=UnifiedRenderGS(render_backend='http',client_url='http://127.0.0.1:8876',scene_id='0226_840298');imgs=asyncio.run(rr.render_tasks(tasks)); rec=[]; thumbs=[]
for i,(r,p,img) in enumerate(zip(rows,poses,imgs)):
 params=dict(r.get('task_params') or {});params.update({'_target_object':r['target_object'],'_camera_intrinsics':eff(r['init_camera']['intrinsics']).tolist(),'_image_width':256,'_image_height':256,'_camera_pose_c2w':p.tolist()});vm=compute_visual_bbox_metrics(p[:3,3],p[:3,2],'projective_relations',params,r['target_region']); q=r['target_region']['params'];a=np.asarray(q['object_a_center']);b=np.asarray(q['object_b_center']);ab=b[:2]-a[:2];ab/=np.linalg.norm(ab);cur=np.array([ab[1],-ab[0]]) if q['relation']=='left' else np.array([-ab[1],ab[0]]); old=np.asarray(q['normal']); old/=np.linalg.norm(old); im=img.copy();d=ImageDraw.Draw(im);d.text((2,2),f'{i} {q["relation"]} ok={vm.get("visual_relation_satisfied")}',fill=(255,255,0));im.save(out/f'{i:03d}.png');thumbs.append(im.resize((128,128)));rec.append({'index':i,'prompt':r['task_description'],'relation':q['relation'],'old_normal':old.tolist(),'current_generator_normal':cur.tolist(),'old_normal_matches_current':bool(np.allclose(old,cur)),'target_pose_c2w':p.tolist(),'visual_metrics':vm})
sheet=Image.new('RGB',(1280,1280),'white');[sheet.paste(x,((i%10)*128,(i//10)*128)) for i,x in enumerate(thumbs)];sheet.save(out/'contact_sheet.png');open(out/'manifest.jsonl','w').write(''.join(json.dumps(x)+'\n' for x in rec));print(json.dumps({'n':len(rec),'target_relation_true':sum(bool(x['visual_metrics'].get('visual_relation_satisfied')) for x in rec),'normal_current_match':sum(x['old_normal_matches_current'] for x in rec)}))
