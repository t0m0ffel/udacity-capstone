from azureml.core import ComputeTarget
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException

def get_compute_cluster(ws):
    cluster_name = "project-cluster"
    vm_size = "Standard_D2_V2"

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                               max_nodes=4
                                                               )
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)
    return compute_target
