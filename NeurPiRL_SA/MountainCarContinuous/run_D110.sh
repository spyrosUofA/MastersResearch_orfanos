
for oracle in {1..1}; do
  for seed in {12222..12222}; do
    o1="2x4/$oracle"
    o2="64x64/$oracle"

    #python3 main.py -time 3660 -seed ${seed} -oracle $o1 -e "DAgger" --bo --aug_dsl -approach "0"
    #python3 main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --bo --aug_dsl -approach "0"
    python3 main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --bo --aug_dsl -approach "0" -c 5000 #-ip "D110/64x64/1/sa_cpus-1_n-100_c-5000_run-122.pkl" -n 25

  done
done