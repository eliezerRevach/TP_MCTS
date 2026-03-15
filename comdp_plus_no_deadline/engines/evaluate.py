from statistics import mean


def evaluation_loop(runs: int, plan_func, params):
    results = [plan_func(*params) for _ in range(runs)]
    success_rate = mean(result.success for result in results)
    avg_makespan = mean(result.makespan for result in results)
    avg_plan_length = mean(result.plan_length for result in results)
    avg_reward = mean(result.cumulative_reward for result in results)
    return {
        "runs": runs,
        "success_rate": success_rate,
        "avg_makespan": avg_makespan,
        "avg_plan_length": avg_plan_length,
        "avg_cumulative_reward": avg_reward,
        "results": results,
    }

