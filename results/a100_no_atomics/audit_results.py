from pathlib import Path
import collections,hashlib,io,json,re,statistics,tarfile
base=Path('results/a100_no_atomics'); r=next(base.glob('*/*/status.txt')).parent
issues=[]; archive=r.with_suffix('.tar.gz')
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert sha==Path(str(archive)+'.sha256').read_text().split()[0]
assert (r/'status.txt').read_text().strip()=='COMPLETE'
assert all(l.split('\t')[1]=='0' for l in (r/'status.tsv').read_text().splitlines())
manifest=json.loads((r/'source_manifest.json').read_text())
with tarfile.open(r/'source.tar.gz') as t:
 src={str(Path(m.name)):t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
assert {n:hashlib.sha256(b).hexdigest() for n,b in src.items()}==manifest['files']
atomic=[n for n,b in src.items() if n.endswith(('.cu','.cuh')) and re.search(rb'\batomic\w*\s*\(',b)]
assert not atomic
rows=[]; data_hashes=set()
for line in (r/'expected_runs.tsv').read_text().splitlines():
 path,mode,seed,batch,profile=line.split('\t'); p=r/'mnist'/Path(path).name
 d=json.loads(p.with_suffix('.json').read_text()); txt=p.read_text()
 want=dict(mode=mode,seed=int(seed),batch=int(batch),profile=bool(int(profile)),epochs=2,slices=3,lr=.01,steps=2*(60000//int(batch)))
 assert all(d.get(k)==v for k,v in want.items()),p
 assert set(d['hashes'])=={'init','weights_epoch_1','weights_epoch_2','test_logits','train_logits'}
 data_hashes.add(re.search(r'^hash data ([0-9a-f]{64})$',txt,re.M)[1])
 d['loss_bits']=re.findall(r'\(bits ([0-9a-f]{8})\)',txt)
 assert len(d['loss_bits'])==d['steps']//100
 rows.append(dict(d,file=p.with_suffix('.json').name))
assert len(rows)==43
other=[]
for p in sorted(Path('bit-repro-results').glob('*/*.tar.gz')):
 with tarfile.open(p) as t:
  for m in t.getmembers():
   if m.isfile() and m.name.endswith('/source_manifest.json'):
    assert json.load(t.extractfile(m))['files']==manifest['files']
   if m.isfile() and '/mnist/' in m.name and m.name.endswith('.json'):
    d=json.load(t.extractfile(m)); txt=t.extractfile(m.name[:-5]+'.txt').read().decode()
    data_hashes.add(re.search(r'^hash data ([0-9a-f]{64})$',txt,re.M)[1])
    d['loss_bits']=re.findall(r'\(bits ([0-9a-f]{8})\)',txt); other.append(d)
assert len(other)==280 and len(data_hashes)==1
comparison=[]
for mode in ['repro','repdl']:
 groups=collections.defaultdict(list)
 for d in rows+other:
  if d['mode']==mode: groups[(d['seed'],d['batch'],d['slices'])].append(d)
 for k,v in sorted(groups.items()):
  count=len({json.dumps(d['hashes'],sort_keys=True) for d in v})
  loss_count=len({tuple(d['loss_bits']) for d in v})
  assert count==loss_count==1,(mode,k)
  comparison.append(dict(mode=mode,seed=k[0],batch=k[1],slices=k[2],records=len(v),distinct_hash_sets=count,distinct_logged_loss_sequences=loss_count))
smoke=json.loads((r/'smoke.json').read_text())
assert smoke['steps']==10 and smoke['hashes']['weights_epoch_1']=='286b280c7e33bcb70aa2bf329f1e01a8e7764765e90f2305f478afdbe895f64d'
summary={}
for mode in ['repro','repdl','stock']:
 ds=[d for d in rows if d['mode']==mode]
 b=[d for d in ds if d['file'].startswith('baseline_')]
 a=[d for d in ds if d['file'].startswith('accuracy_') or d['file']==f'baseline_{mode}_repeat1.json']
 assert len(b)==3 and len(a)==5, (mode,len(b),len(a))
 summary[mode]=dict(baseline_step_ms=statistics.median(d['step_ms_median'] for d in b),baseline_train_seconds=statistics.median(d['train_seconds'] for d in b),
   baseline_step_ms_runs=[d['step_ms_median'] for d in b],test_accuracy_mean=statistics.mean(d['test_accuracy'] for d in a),
   test_accuracy_sd=statistics.stdev(d['test_accuracy'] for d in a),peak_allocated_memory_mib=statistics.median(d['peak_memory_mib'] for d in b),
   stage_ms_median={k:statistics.median(d['breakdown_ms'][k] for d in ds if d['profile']) for k in ['forward','backward','optimizer']},
   scaling_step_ms={str(batch):statistics.median(d['step_ms_median'] for d in ds if d['batch']==batch) for batch in [50,200,500,1000] if any(d['batch']==batch for d in ds)})
profiles={}
for name,expected in [('repro_sample',10),('repro_conv_gemm',12),('wgrad_batch100',2),('wgrad_batch500',2)]:
 text=(r/'profiling'/f'{name}.txt').read_text()
 durations=[float(value.replace(',',''))*(1000 if unit in ('ms','msecond') else 1) for unit,value in re.findall(r'^\s*Duration\s+(us|usecond|ms|msecond)\s+([\d.,]+)',text,re.M)]
 assert len(durations)==expected,(name,len(durations))
 p=r/'profiling'/f'{name}.ncu-rep'; assert p.stat().st_size>0
 profiles[name]=dict(kernel_durations_us=durations,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
old_matches=0
with tarfile.open('a100_atomic_results.tar') as t:
 for m in t.getmembers():
  if not m.isfile() or not m.name.endswith('.json'):continue
  d=json.load(t.extractfile(m))
  if d.get('mode')!='repro':continue
  match=[x for x in rows if x['mode']=='repro' and all(x.get(k)==d.get(k) for k in ['seed','batch','slices','epochs','steps','lr'])]
  if match:
   assert all(x['hashes']==d['hashes'] for x in match),m.name
   old_matches+=1
pids=re.findall(r'^\s*Process ID\s*:\s*(\d+)',(r/'environment/gpu.txt').read_text(),re.M)
result=dict(run=str(r),archive_sha256=sha,checks_passed=True,full_runs=43,source_files=len(src),cuda_files=sum(n.endswith(('.cu','.cuh')) for n in src),
  source_matches_all_cims_archives=True,explicit_cuda_atomic_candidates=atomic,software=dict(torch=rows[0]['torch'],torch_cuda=rows[0]['cuda'],nvcc='12.6.77',driver=rows[0]['driver']),
  combined_configurations=8,combined_full_runs=323,cross_gpu_groups=comparison,matching_atomic_a100_records=old_matches,
  summary=summary,profiles=profiles,preexisting_gpu_pids=pids,
  caveats=['A pre-existing python3 process (PID 7206, 420 MiB) was recorded on the GPU at setup; active compute overlap is not established.',
   'PyTorch wheel CUDA 12.4 and extension compiler CUDA 12.6 constitute one mixed build configuration.',
   'Source scan is not an audit of generated instructions or third-party libraries.',
   'Archive payloads/checksum and profiler text exports were checked; binary .ncu-rep files were not opened in Nsight locally.',
   'The extracted runner.log has final archive/status messages appended after the archive was created; all other archive files match the supplied directory.'])
(base/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['A100 NON-ATOMICS RESULTS AUDIT','',str(r),'PASS: supplied archive SHA256 matches sidecar; 43 full runs, smoke, all command exits and four profiler captures checked.',
 f'PASS: {len(src)} archived source files match their manifest and all four CIMS source snapshots; 35 CUDA files have no explicit atomic call candidates.',
 'PASS: repro (131 combined records, 9 setting groups) and RepDL (64 records, 5 seed groups) match across A100 and the seven CIMS configurations, including all recorded hashes and logged loss bit sequences.',
 f'PASS: {old_matches} comparable old atomic A100 repro records match the new A100 results.',
 'Eight software/hardware configurations and 323 full training runs in total, across four GPU models.',
 '', 'Configuration: A100-SXM4-40GB, PyTorch 2.6.0+cu124, nvcc 12.6.77, Python 3.9 extension, driver '+rows[0]['driver']+'.',
 'This is ONE mixed CUDA build configuration, not a CUDA version sweep.',
 '', 'Baseline timing: median of three unprofiled seed-0 batch-100 run medians. Accuracy: mean over seeds 0..4, using baseline repeat 1 for seed 0.',
 'Mode       Step ms    Test accuracy mean    Peak allocated memory MiB']
for mode,s in summary.items():lines.append(f'{mode:8} {s["baseline_step_ms"]:9.3f} {100*s["test_accuracy_mean"]:20.3f}% {s["peak_allocated_memory_mib"]:27.1f}')
lines += ['',f'Recorded RepDL/repro time ratio: {summary["repdl"]["baseline_step_ms"]/summary["repro"]["baseline_step_ms"]:.2f}x; repro/stock time ratio: {summary["repro"]["baseline_step_ms"]/summary["stock"]["baseline_step_ms"]:.2f}x.',
 'Performance caveat: pre-existing python3 PID 7206 held 420 MiB at setup. This does not establish active overlap, but prevents claiming an exclusive/idle GPU benchmark from the saved snapshot alone.',
 'Repro stage medians across four profile runs: '+str(summary['repro']['stage_ms_median']),
 '', 'Profiler weight-gradient duration, batch 100: 397.38 us and 655.33 us; batch 500: 1.98 ms and 3.27 ms.',
 'The basic launch sample now captures convolution/reduction/pooling/finite-scan work, not the same Adam-dominated kernels as the earlier atomic sample. Launch offsets do not identify the same operations after implementation changes.',
 '',*result['caveats'],'','Detailed batch scaling, profiler hashes/durations and group checks: audit_summary.json.']
(base/'AUDIT_RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
