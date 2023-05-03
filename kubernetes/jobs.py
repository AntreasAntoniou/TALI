# Read script template, make per experiment modifications
# For each experiment script, create a job kubernetes spec file

import copy
import subprocess
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import pkg_resources as pkg
import tqdm
import yaml
from rich import print


@dataclass
class ExperimentTemplate:
    standard: str = "job.template.yaml"
    standard_pd: str = "job-with-pdisk.template.yaml"


class Job(object):
    def __init__(
        self,
        name: str,
        script_list: List[str],
        docker_image_path: str,
        secret_variables: List[tuple] = None,
        environment_variables: Dict[str, str] = None,
        num_repeat_experiment: int = 5,
        experiment_template: str = ExperimentTemplate.standard,
        shm_size: str = "80Gi",
        kubernetes_spec_dir: Union[str, Path] = Path(
            "generated/kubernetes/specs"
        ),
        persistent_disk_claim_names_to_mount_dict: Dict[str, str] = None,
        multi_persistent_disk_claim_names_to_mount_dict: Dict[
            str, Dict[str, str]
        ] = None,
        num_gpus: int = 1,
        image_pull_policy: str = "Always",
    ):
        # to add additional features you might find these pages useful
        # https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/job-v1/#JobSpec
        # https://kubernetes.io/docs/concepts/workloads/controllers/job/
        self.name = name
        self.script_list = script_list
        self.environment_variables = environment_variables
        self.secret_variables = secret_variables
        self.container_path = docker_image_path
        self.num_repeat_experiment = num_repeat_experiment
        self.experiment_template = experiment_template
        self.shm_size = shm_size
        self.kubernetes_spec_dir = Path(kubernetes_spec_dir)
        self.spec_file_list = None
        self.spec_dict_list = None
        self.gen_idx = 0
        self.persistent_disk_claim_names_to_mount_dict = (
            persistent_disk_claim_names_to_mount_dict
        )
        self.multi_persistent_disk_claim_names_to_mount_dict = (
            multi_persistent_disk_claim_names_to_mount_dict
        )
        self.num_gpus = num_gpus
        self.image_pull_policy = image_pull_policy

        if kubernetes_spec_dir == ExperimentTemplate.standard_pd:
            self.use_persistent_disks = True
            if persistent_disk_claim_names_to_mount_dict == None:
                raise ValueError(
                    "For persistent disk experiment templates you must \
                define a persistent_disk_claim_names_to_mount_dict variable consisting \
                of persistent_disk_claims to mounting directories for an instance"
                )
        else:
            self.use_persistent_disks = False

        if not self.kubernetes_spec_dir.exists():
            self.kubernetes_spec_dir.mkdir(parents=True)

    def generate_spec_files(self):
        spec_template = Path(
            pkg.resource_filename(
                __name__, f"../templates/{self.experiment_template}"
            )
        )
        print(f"Using spec template: {spec_template}")
        spec_dict = yaml.safe_load(spec_template.read_text())
        spec_dict["spec"]["template"]["spec"]["containers"][0][
            "name"
        ] = "job-container"
        spec_dict["spec"]["template"]["spec"]["containers"][0][
            "image"
        ] = self.container_path

        spec_dict["spec"]["backoffLimit"] = self.num_repeat_experiment
        spec_dict["spec"]["template"]["spec"]["volumes"][0]["emptyDir"][
            "sizeLimit"
        ] = self.shm_size
        spec_dict["spec"]["template"]["spec"]["containers"][0]["resources"][
            "limits"
        ]["nvidia.com/gpu"] = self.num_gpus

        spec_dict["spec"]["template"]["spec"]["containers"][0][
            "imagePullPolicy"
        ] = self.image_pull_policy

        volume_claims = []
        volume_mounts = []

        for (
            pvc_name,
            job_mount_dir,
        ) in self.persistent_disk_claim_names_to_mount_dict.items():
            volume_claims.append(
                dict(
                    name=f"{pvc_name}-vol",
                    persistentVolumeClaim=dict(
                        claimName=pvc_name, readOnly=True
                    ),
                )
            )
            volume_mounts.append(
                dict(
                    mountPath=job_mount_dir,
                    name=f"{pvc_name}-vol",
                    readOnly=True,
                )
            )

        spec_dict["spec"]["template"]["spec"]["volumes"].extend(volume_claims)
        spec_dict["spec"]["template"]["spec"]["containers"][0][
            "volumeMounts"
        ].extend(volume_mounts)

        spec_dict_list = []
        for idx, script_entry in enumerate(self.script_list):
            current_dict = copy.deepcopy(spec_dict)
            current_dict["metadata"]["name"] = f"{self.name}-{idx}"
            current_dict["spec"]["template"]["spec"]["containers"][0][
                "command"
            ] = list(script_entry.split(" "))

            env_variables_list = []

            if self.secret_variables is None:
                self.secret_variables = {}

            for (
                env_variable_name,
                context_name,
            ) in self.secret_variables.items():
                env_variables_list.append(
                    {
                        "name": env_variable_name,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": context_name,
                                "key": env_variable_name,
                            }
                        },
                    },
                )

            if self.environment_variables is None:
                self.environment_variables = {}

            env_variables_list.extend(
                {"name": key, "value": value}
                for key, value in self.environment_variables.items()
            )

            current_dict["spec"]["template"]["spec"]["containers"][0][
                "env"
            ] = env_variables_list

            volume_claims = []
            volume_mounts = []

            for (
                pvc_multi_name,
                job_mount_dir,
            ) in self.multi_persistent_disk_claim_names_to_mount_dict.items():
                pvc_names = list(job_mount_dir.keys())[
                    idx % len(job_mount_dir)
                ]
                pvc_path = list(job_mount_dir.values())[
                    idx % len(job_mount_dir)
                ]
                volume_claims.append(
                    dict(
                        name=f"{pvc_names}-vol",
                        persistentVolumeClaim=dict(
                            claimName=pvc_names, readOnly=False
                        ),
                    )
                )
                volume_mounts.append(
                    dict(
                        mountPath=pvc_path,
                        name=f"{pvc_names}-vol",
                        readOnly=False,
                    )
                )

            current_dict["spec"]["template"]["spec"]["volumes"].extend(
                volume_claims
            )
            current_dict["spec"]["template"]["spec"]["containers"][0][
                "volumeMounts"
            ].extend(volume_mounts)

            spec_dict_list.append(current_dict)

        spec_file_list = []
        for idx, spec_dict in enumerate(spec_dict_list):
            spec_file = self.kubernetes_spec_dir / f"{self.name}-{idx}.yaml"
            with open(spec_file, "w+") as spec_fp:
                yaml.safe_dump(spec_dict, spec_fp)

            spec_file_list.append(spec_file)

        self.spec_file_list = spec_file_list
        self.spec_dict_list = spec_dict_list
        self.gen_idx += 1

    def run_jobs(self):
        output_dict = {}
        with tqdm.tqdm(total=len(self.spec_file_list)) as pbar:
            for script_file in self.spec_file_list:
                result = subprocess.run(
                    f"kubectl create -f {script_file.as_posix()}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                output_dict[script_file] = {
                    "stdout": result.stdout.decode("utf-8").split("\n"),
                    "stderr": result.stderr.decode("utf-8").split("\n"),
                }
                pbar.update(1)
                pbar.set_description(f"Running {script_file.name}")
        return output_dict


if __name__ == "__main__":
    from quote import quote

    quotes = quote("hume", limit=10)

    script_list = [
        f"echo {quote_instance['quote']}" for quote_instance in quotes
    ]

    exp = Job(
        name="dummy-exp",
        script_list=script_list,
        docker_image_path="ghcr.io/bayeswatch/compute-gpu:0.1.0",
        num_repeat_experiment=3,
    )

    exp.generate_spec_files()
    output = exp.run_jobs()
    print(output)
