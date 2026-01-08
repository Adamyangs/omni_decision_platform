for i in {1..7}
do
    python3 src/main.py --config=perla_mappo --env-config=sc2 with env_args.map_name=3s_vs_5z &
    python3 src/main.py --config=qscan --env-config=sc2 with env_args.map_name=3s_vs_5z &
    python3 src/main.py --config=rode --env-config=sc2 with env_args.map_name=3s_vs_5z &
    wait
done

