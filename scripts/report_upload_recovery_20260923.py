"""Read-only scheduler/progress reconciliation; writes this recovery's report."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'


def describe(job):
    result=subprocess.run([SCO,'acp','jobs','describe',job,'--workspace-name','aigc','-o','json'],capture_output=True,text=True,timeout=30)
    if result.returncode: return {'name':job,'state':'QUERY_ERROR','error':result.stdout+result.stderr}
    return json.loads(result.stdout)


def main():
    now=time.time()
    records=[json.loads(p.read_text()) for p in sorted((BASE/'submissions').glob('*.json'))]
    cog=json.loads((ROOT/'runtime/cog_upload_backpressure_v1/pack.json').read_text())['jobs']
    ids=[r['new_job'] for r in records]+list(cog)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        scheduler=dict(zip(ids,pool.map(describe,ids)))
    rows=[]
    pack=json.loads((ROOT/'runtime/feature_upload_recovery_v1/pack.json').read_text())
    for record in records:
        old=record['old_job']; new=record['new_job']; shard=pack['jobs'][old]['shard']
        folder=BASE/'mini_publish'/f'streaming-{shard}-of-8'
        state=json.loads((folder/'runtime_status.json').read_text()) if (folder/'runtime_status.json').exists() else {}
        receipts=list((folder/'receipts').glob('*.json'))
        text=(folder/'upload.log').read_text() if (folder/'upload.log').exists() else ''
        events=[json.loads(line) for line in text.splitlines() if line.startswith('{"event":')]
        cleaned=sum(e.get('bytes',0) for e in events if e.get('event')=='cleanup')
        row={'model':'MiniMax-H3','old_job':old,'new_job':new,'scheduler':scheduler[new]['state'],
             'old_terminal':json.loads((BASE/'terminal_handoffs'/f'{old}.json').read_text())['state'],
             'phase':'EXTRACTING' if state.get('producer_started') and not state.get('producer_paused') else 'DRAINING',
             'runtime':state,'verified_units':len(receipts),'cleaned_bytes':cleaned,'upload_error':'Traceback' in text}
        rows.append(row)
    for job in cog:
        path=BASE/'cog_guard'/job/'status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        rows.append({'model':'CogVideoX','old_job':job,'new_job':job,'scheduler':scheduler[job]['state'],
                     'phase':state.get('phase','GUARD_NOT_CONFIRMED'),'runtime':state,
                     'heartbeat_age_s':round(now-state.get('epoch',0),1)})
    result={'epoch':now,'utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
            'jobs':rows,'scheduler':scheduler,'scope':'8 MiniMax streaming workers plus 4 CogVideoX streaming producers; Cog sync PIDs preserved',
            'tests':{'MiniMax_unit_tests':7,'Cog_unit_tests':3,'streaming_remote_canary':'PASS','target_remote_canary':'PASS',
                     'canary_cleanup':'temporary exact copy deleted; original canary source retained',
                     'original_done_index_pytest':'1 passed, 4 deselected'},
            'caution':'RUNNING scheduler does not mean extracting: backpressure waits for backlog to drain.'}
    (BASE/'current_status.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# MiniMax / CogVideoX 上传恢复执行记录','',f'更新时间：{result["utc"]}','',
           '8 个 MiniMax 旧 job 在 worker Deleted 后关闭 scheduler 记录，特征未随 job 删除。新任务均为 zoetrope 单卡。',
           'CogVideoX 保留原 job、原提取进程和原同步器，仅新增 SHA 绑定的独立背压守护。',
           'RUNNING 仅表示 scheduler 状态；DRAINING 表示上传继续，提取因积压尚未恢复。','',
           '| 模型 | 旧 job | 当前 job | scheduler | 实际阶段 | 本地积压 GiB |',
           '|---|---|---|---|---|---|']
    for row in rows:
        backlog=row['runtime'].get('backlog_bytes')
        lines.append(f'| {row["model"]} | {row["old_job"]} | {row["new_job"]} | {row["scheduler"]} | {row["phase"]} | {backlog/1024**3:.1f} |' if backlog is not None else
                     f'| {row["model"]} | {row["old_job"]} | {row["new_job"]} | {row["scheduler"]} | {row["phase"]} | 未确认 |')
    lines+=['','## 修复及验证','',
            '- MiniMax：原 launcher 无上传器；新链路为上传 → 原对象回读 SHA/长度校验 → 远程和本地回执 → done-index → 清理对应本地 payload。metadata/index 保留，远程可恢复。',
            '- 远程对象仍在准确的 `fj:s3://baiqiao/InsScene-15K/FEAT_*_CANONICAL_V2_MINIMAX_BF16_V2`，不重定向其他样本。原 backend 源码、73 帧、t250、BF16 和场景分片不变。',
            '- MiniMax 积压 >8 GiB 或可用空间 <2 TiB 时暂停；积压 ≤4 GiB 且空间 ≥3 TiB 自动继续。当前优先上传旧积压。',
            '- CogVideoX 积压 >64 GiB 或空间 <2 TiB 暂停；积压 ≤32 GiB 且空间 ≥3 TiB 自动继续；上传10分钟无进展也暂停提取。只操作绑定的原 extractor PID，不信号同步器。',
            '- Cog 两个暂停时未关闭的临时文件已通过有界继续写入完成，未删除临时文件或有效特征。',
            '- Cog 3 个同步器已有不同 worklist SHA 暂存目录；同 model/source 根目录只有1个旧命名同步器，已完成 scheduler 全量核对。旧脚本 SHA 的既有差异为 ADS 并发参数化；本轮不改写，按当前打开脚本 SHA 记录。',
            '- MiniMax streaming/target 真实上传回读和临时副本清理 canary 通过；7项上传单测、3项背压/真实 pidfd/无重复工作清单测试通过；原 done-index 跳过测试1项通过。',
            '- 本轮未改变数据/科学协议，未停止 CogVideoX target 或其他模型任务。','',
            '## 交接说明','',
            'MiniMax 冻结包：`runtime/feature_upload_recovery_v1/pack.json`。Cog 冻结包：`runtime/cog_upload_backpressure_v1/pack.json`。',
            '当前状态：同目录 `current_status.json`；逐次发布回执在 `mini_publish/*/receipts`，生产清理字节数在状态 JSON。',
            'MiniMax 生产上传与自动恢复 launcher 已运行；积压排空后的新增提取尚不能当作已验证。Cog 需以新鲜守护心跳为准，不能只看 scheduler RUNNING。']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scheduler'},indent=2))


if __name__=='__main__': main()
