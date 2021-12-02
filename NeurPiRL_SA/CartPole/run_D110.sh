
for oracle in {11..11}; do
  for seed in {1..1}; do
    o1="2x4/$oracle"
    o2="64x64/$oracle"

    #python3 main.py -time 3660 -seed ${seed} -oracle $o1 -e "DAgger" --bo --aug_dsl -approach "0"
    #python3 main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --bo --aug_dsl -approach "0"
    python3 main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --bo -approach "0"

  done
done



