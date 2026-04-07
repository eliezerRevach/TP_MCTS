# CoMDP+ No-Deadline Starter

This folder is a minimal starter environment for studying CoMDP+ with:

- probabilistic outcomes
- durative/concurrent actions
- STN consistency
- no global deadline constraint

It keeps the original thesis domain logic and removes deadline setting in selected scenarios.

## Included starter scenarios

- `easy_stuck_car_1o`
- `easy_nasa_rover_1`
- `mid_nasa_rover_2`
- `mid_machine_shop_2`
- `hard_nasa_rover_3`

## Run greedy baseline

From repository root:

```bash
python -m comdp_plus_no_deadline.run_no_deadline --scenario easy_nasa_rover_1 --domain_type combination --runs 5 --max_steps 250 --seed 123
```

You can also choose domain directly:

```bash
python -m comdp_plus_no_deadline.run_no_deadline --domain nasa_rover --object_amount 2 --domain_type regular --runs 3
```

To use the optimistic probabilistic RPG heuristic:

```bash
python -m comdp_plus_no_deadline.run_no_deadline --domain nasa_rover --domain_type regular --heuristic_name probabilistic_rpg --heuristic_aggregation product --heuristic_layers 25
```

## Run smoke benchmarks (all 5 presets)

```bash
python -m comdp_plus_no_deadline.run_smoke
```

## Run tests for this starter package

```bash
python -m comdp_plus_no_deadline.tests.test_no_deadline_setup
python -m comdp_plus_no_deadline.tests.test_greedy_solver
python -m comdp_plus_no_deadline.tests.test_probabilistic_rpg
```

