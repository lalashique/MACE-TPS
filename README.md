# plan
1. WORKFLOW - mlp_path/workflow
   1. protocol runnner 
   2. clean up between step
   3. logger
2. DFT - mlp_path/dft
   1. Class for setting up DFT calculators
   2. Class for running DFT calculation
3. MLP - mlp_path/mlp
   1. setting up MLP calculators
   2. running mlp calculations
   3. evaluating mlp 
      1. committee
      2. rmsd with dft
4. TPS   - mlp_path/tps
   1. setting up TPS methods
   2. Sampler
5. Util
   1. CV
      1. CV dataclass
      2. CV static functions
   2. parser
      1. functions for reading configs
   3. mlp-helper
      1. god knows