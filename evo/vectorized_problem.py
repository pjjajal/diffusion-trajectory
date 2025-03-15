import torch
import numpy as np
from evotorch import Problem, SolutionBatch
from evotorch.core import SolutionBatchPieces


### Custom Problem Class
class VectorizedProblem(Problem):
    def __init__(
        self,
        objective_sense,
        objective_func=None,
        initial_bounds=None,
        bounds=None,
        solution_length=None,
        dtype=None,
        eval_dtype=None,
        device=None,
        eval_data_length=None,
        seed=None,
        num_actors=None,
        actor_config=None,
        num_gpus_per_actor=None,
        num_subbatches=None,
        subbatch_size=None,
        store_solution_stats=None,
        vectorized=None,
        splits=1,
        initialization=None
    ):
        super().__init__(
            objective_sense,
            objective_func,
            initial_bounds=initial_bounds if initialization is None else None,
            bounds=bounds,
            solution_length=solution_length,
            dtype=dtype,
            eval_dtype=eval_dtype,
            device=device,
            eval_data_length=eval_data_length,
            seed=seed,
            num_actors=num_actors,
            actor_config=actor_config,
            num_gpus_per_actor=num_gpus_per_actor,
            num_subbatches=num_subbatches,
            subbatch_size=subbatch_size,
            store_solution_stats=store_solution_stats,
            vectorized=vectorized,
        )
        self.initialization = initialization
        self.splits = splits

    def _fill(self, values):
        if self.initialization is not None:
            self.initialization(values)
            return values
        return super()._fill(values)

    def _evaluate_batch(self, solutions: SolutionBatch) -> None:
        self._evaluate_batch_override_split_k(solutions)

    def _evaluate_batch_override_split_k(self, solutions: SolutionBatch) -> None:
        ### Baseline implementation - behaves like default
        # i.e., it evaluates each solution sequentially
        # for solution in solutions:
        #     self._evaluate(solution)

        ### Slice the solutions
        solution_batch_splits = SolutionBatchPieces(
            batch=solutions, num_pieces=self.splits
        )
        split_all_result_list = []

        ### Compute a few at a time
        for iteration_idx in range(len(solution_batch_splits)):
            solution_split = solution_batch_splits[iteration_idx]
            split_result_tensor = self._objective_func(solution_split.values)
            #   print(f"split_result_tensor {split_result_tensor}")
            split_all_result_list.append(split_result_tensor)
        split_all_result_tensor = torch.cat(split_all_result_list, dim=0)

        # print(f"split_all_result_tensor.shape {split_all_result_tensor.shape}")
        solutions.set_evals(split_all_result_tensor)
