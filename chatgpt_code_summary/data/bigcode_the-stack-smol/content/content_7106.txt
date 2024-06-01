from tatau_core.models import TaskDeclaration
from tatau_core.models.task import ListEstimationAssignments
from tatau_core.utils.ipfs import Directory


class Estimator:
    @staticmethod
    def get_data_for_estimate(task_declaration):
        dataset = task_declaration.dataset
        ipfs_dir = Directory(dataset.train_dir_ipfs)
        dirs, files = ipfs_dir.ls()
        return {
            'chunk_ipfs': dirs[0].multihash,
            'model_code_ipfs': task_declaration.train_model.code_ipfs,
        }

    @staticmethod
    def estimate(task_declaration: TaskDeclaration, finished_assignments: ListEstimationAssignments):
        failed = False

        assert len(finished_assignments)

        sum_tflops = 0.0
        for estimation_assignment in finished_assignments:
            sum_tflops += estimation_assignment.estimation_result.tflops
            if estimation_assignment.estimation_result.error is not None:
                failed = True
                return 0.0, failed

        av_tflops = sum_tflops / len(finished_assignments)
        ipfs_dir = Directory(task_declaration.dataset.train_dir_ipfs)
        dirs, files = ipfs_dir.ls()
        chunks_count = len(dirs)
        return av_tflops * chunks_count * task_declaration.epochs, failed


