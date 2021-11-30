
for oracle in {5..5}; do
  for seed in {1..1}; do
    o1="2x4/$oracle"
    o2="64x64/$oracle"

    #python3 main.py -time 3660 -seed ${seed} -oracle $o1 -e "DAgger" --aug_dsl -approach "0"
    python3 main.py -time 3660 -seed ${seed} -oracle $o2 -e "DAgger" --aug_dsl --bo -approach "0" -c 5000

  done
done