# Environment

Install torch geometric manually https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
Install dgl manually https://www.dgl.ai/pages/start.html

Then install all further requiremets with

```pip install -r requirements.txt```

# Usage

The `main.py` script is used for all experiments, including dataset generation and model training. Below is an example command to run a quick test with a subset of the data (100 users).

```python3 main.py --dataset beauty train --nologger  --accelerator cpu --epochs 2 --num_loader_workers 0 CTGR --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1 --num_users 100```

Take care to include all the command line arguments.

The first time a dataset is needed, as specified by the part `neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1 --num_users 100`, the script will run the sampling algorithm and store the generated dataset in `precomputed_data`. The next time the same dataset is needed it will reload it from there. You can delete the `precomputed_data` folder if you need to re-generate a dataset with the same configuration.

We make use of a script that cleans up the output of exceptions and puts you in a debugger session whenever one happens. You can exit the debugger with `Ctrl+D` or you can disable it entirely by setting an environment variable with `export NOCOOLTRACEBACK=1`

# Experiments

## Dataset generation
The commands used to generate the datasets are in `generate_dgsr_dataset0.job`, `generate_dgsr_dataset1.job`, `generate_dgsr_dataset2.job`, each with a different seed (randomness applies mainly to the generation of the false candidates in the test set). It can take upwards to 20 hours to generate a single dataset.

## Model training and evaluating
The commands used for the experiments in the report are in the bash scripts in the `/jobs` folder. Most models took about 5 hours to train (see the #Resources section for more info)

## Testing the use of temporal information

To verify if the model is utilizing the time information, or if it solely benefits from the number of parameters you can evaluate the model with random time.

To do this, simply use the command:
```python main.py random_test --load_checkpoint CTGRLOD-30```

where you replace CTGRLOD-30 with the name of your Neptune run

This will add an additional namespace to the checkpoint called test_RANDOM with an evaluation on data with randomized time.

Additional parameters include: `--testnormaltoo` and `--batch_size`

To run this test for all experiments in neptune, edit `jobs/randomized_time.sh` replacing `CTGR` with the namespace you used and `188` with the number of runs you had. Then run `export NOCOOLTRACEBACK=1; bash jobs/randomized_time.sh`

# Logging

We used the neptune logger for our experiments. If you wish to enable logging, create a file `keys.sh` with the content

```
export NEPTUNE_PROJECT='{neptune_project}'
export NEPTUNE_API_TOKEN="{neptune_key}"
```

And source it with ```source keys.sh``` before running.

Logging is required for checkpointing, so if it is disabled, the last model will be used for testing rather than the model with the highest validation score.

# DGSR-reproduction

Reusable python code is in sgat

Voor de rest gebruik root voor notebooks en executeerbaare scripys


# Resources

GPU

```
lcur1393@r32n2:~/CONT$ nvidia-smi
Sun Jun  5 14:07:32 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:5E:00.0 Off |                  N/A |
|  0%   23C    P8     7W / 250W |      1MiB / 11178MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

CPU

```
lcur1393@r32n2:~/CONT$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
Address sizes:       46 bits physical, 48 bits virtual
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  1
Core(s) per socket:  6
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Xeon(R) Bronze 3104 CPU @ 1.70GHz
Stepping:            4
CPU MHz:             800.081
BogoMIPS:            3400.00
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            8448K
NUMA node0 CPU(s):   0,2,4,6,8,10
NUMA node1 CPU(s):   1,3,5,7,9,11
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti intel_ppin ssbd mba ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm arat pln pts pku ospke md_clear flush_l1d arch_capabilities
```

RAM
```
lcur1393@r32n2:~/CONT$ cat /proc/meminfo
MemTotal:       263772360 kB
MemFree:        239230336 kB
MemAvailable:   252454264 kB
Buffers:            1220 kB
Cached:         16161184 kB
SwapCached:          104 kB
Active:          9443300 kB
Inactive:       13584252 kB
Active(anon):    7576392 kB
Inactive(anon):   905960 kB
Active(file):    1866908 kB
Inactive(file): 12678292 kB
Unevictable:           0 kB
Mlocked:               0 kB
SwapTotal:      16777212 kB
SwapFree:       16756208 kB
Dirty:                20 kB
Writeback:             0 kB
AnonPages:       6858244 kB
Mapped:          1256080 kB
Shmem:           1633436 kB
Slab:             633208 kB
SReclaimable:     345524 kB
SUnreclaim:       287684 kB
KernelStack:        6576 kB
PageTables:        27852 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    277911848 kB
Committed_AS:   10010792 kB
VmallocTotal:   34359738367 kB
VmallocUsed:           0 kB
VmallocChunk:          0 kB
Percpu:            35648 kB
HardwareCorrupted:     0 kB
AnonHugePages:   6383616 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:    13645404 kB
DirectMap2M:    250212352 kB
DirectMap1G:     6291456 kB

```
