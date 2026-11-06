# Cu_CO2 example data (copied 2026-11-06 from the reference Cu_CO25 run)

Copies, originals untouched. data/*.gz = gzip -9 of reference <gen>/mlp/data/{train,test}.xyz (cumulative). models = <gen>/mlp/runs/c*/MACE_models/mace01.model. smd/path0|path5 = ref smd/0/1 and smd/5/1 (coord.xyz -> smd.xyz). dft.json/mlp.yaml/tps.json/top.pdb/tps.xyz derived from the reference run. Gen 3 is incomplete in the reference.

```
007f53e2ec60c0b5       6.3 MB  ./data/gen0/test.xyz.gz
063b3891206eee69      25.3 MB  ./data/gen0/train.xyz.gz
d4535c51b86f9e29       8.7 MB  ./data/gen1/test.xyz.gz
dfe563889c7fb6bd      34.5 MB  ./data/gen1/train.xyz.gz
ab1aae339a89593c      11.2 MB  ./data/gen2/test.xyz.gz
34d589e079ee7900      44.5 MB  ./data/gen2/train.xyz.gz
18834d0cd4ec3bad       0.0 MB  ./dft.json
bb5854d51264c563      13.2 MB  ./init/coord1.xyz
3975f230a1c551a1       6.6 MB  ./init/coord10.xyz
97cd27859b42fcf8      14.5 MB  ./init/coord11.xyz
663257a1c75fd815      13.3 MB  ./init/coord12.xyz
c3bc9ec8dfdaa255      13.3 MB  ./init/coord13.xyz
15ba703a6ebf8d7e      13.3 MB  ./init/coord2.xyz
e668dffe7b311408      13.2 MB  ./init/coord3.xyz
a38b263d3afd15a3      13.3 MB  ./init/coord4.xyz
b7c8ab8bcf310622      13.2 MB  ./init/coord5.xyz
e99081fdcd8b8d4b      13.3 MB  ./init/coord6.xyz
7c102b2a8fe9354f       6.7 MB  ./init/coord7.xyz
93c37d4929685851       6.7 MB  ./init/coord8.xyz
d7ba03a92ec62b14       6.6 MB  ./init/coord9.xyz
d1232136a2e3a7fe       0.0 MB  ./init/cp2k.inp
88e00ad1c43c8927       0.0 MB  ./init/force_eval.inp
a128bc75797ca9b2       0.0 MB  ./mlp.yaml
a92af2964203ccb5       7.7 MB  ./models/gen0/c0/mace01.model
3e0f3e6f6b202b38       7.7 MB  ./models/gen0/c1/mace01.model
f3b96a50610e9dd6       7.7 MB  ./models/gen0/c2/mace01.model
17dfcffe60d5772a       7.7 MB  ./models/gen1/c0/mace01.model
7b5d4b7fa07b2805       7.7 MB  ./models/gen1/c1/mace01.model
6cd80bce6214f2ab       7.7 MB  ./models/gen1/c2/mace01.model
418a1fcb8ae8d120       7.7 MB  ./models/gen2/c0/mace01.model
ae45772f09b2ab8c       7.7 MB  ./models/gen2/c1/mace01.model
f511b361a551865d       7.7 MB  ./models/gen2/c2/mace01.model
6b73c1fe68462a7c       7.7 MB  ./models/gen3/c0/mace01.model
667b4deea5a15338       7.7 MB  ./models/gen3/c1/mace01.model
d3866533b130f07c       7.7 MB  ./models/gen3/c2/mace01.model
b4d7cddee4529ecb       0.0 MB  ./smd/path0/plumed.dat
5d3842f53fd44df4       0.0 MB  ./smd/path0/smd.json
781682df08b9e799       0.0 MB  ./smd/path0/smd.xyz
b4d7cddee4529ecb       0.0 MB  ./smd/path5/plumed.dat
5d3842f53fd44df4       0.0 MB  ./smd/path5/smd.json
781682df08b9e799       0.0 MB  ./smd/path5/smd.xyz
7a3bb1057be98735       0.0 MB  ./top.pdb
3967b313ffbe30e7       0.0 MB  ./tps.json
8a67567a7ecb6d9e      15.9 MB  ./tps.xyz
```

cp2k_data/: CP2K data files named in init/force_eval.inp, copied from CP2K 2026.1 (Homebrew). The reference run used CP2K 2024.1 - compare against your install before relying on them. Use: export CP2K_DATA_DIR=$PWD/cp2k_data (CP2K is GPL-2.0; check redistribution).
96070872512da557      43.6 MB  ./data/gen3/train.xyz.xz
0576e2a36bccad5d      11.0 MB  ./data/gen3/test.xyz.xz
