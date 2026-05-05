/home/dat.tt2/miniconda3/envs/dg/bin/python \
  domainbed/scripts/make_routing_table.py \
  --routing_raw train_output/routing_diagnostics/L_inv_env3/routing_raw.npz \
  --method GMOE_InvMMD \
  --output_dir train_output/routing_diagnostics/L_inv_env3/table
