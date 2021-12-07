
for oracle in {1..2}; do
  for seed in {1..2}; do
    o1="256x0/$oracle"
    o2="64x64/$oracle"

    python3 main.py -time 7260 -seed ${seed} -oracle $o1 -e "DAgger" --bo --aug_dsl -approach "0"
    python3 main.py -time 7260 -seed ${seed} -oracle $o2 -e "DAgger" --bo --aug_dsl -approach "0"
    #python3 main.py -time 7260 -seed ${seed} -oracle $o2 -e "DAgger" --bo  -approach "0" #-c 5000 #-ip "D110/64x64/1/sa_cpus-1_n-100_c-5000_run-21.pkl"
  done
done