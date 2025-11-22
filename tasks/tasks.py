from tasks.qa import TaskQA
from tasks.pi import TaskPI
from tasks.sa import TaskSA
from tasks.td import TaskTD
from tasks.nli import TaskNLI
from tasks.cr import TaskCR
from tasks.asu import TaskAS
from tasks.mt import TaskMT

def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("qa"):
        return TaskQA(**kwargs)
    elif task_name == "pi":
        return TaskPI(**kwargs)
    elif task_name == "sa":
        return TaskSA(**kwargs)
    elif task_name == "td":
        return TaskTD(**kwargs)
    elif task_name == "nli":
        return TaskNLI(**kwargs)
    elif task_name == "cr":
        return TaskCR(**kwargs)
    elif task_name.startswith("asu"):
        return TaskAS(**kwargs)
    elif task_name.startswith("mt"):
        return TaskMT(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")

if __name__ == "__main__":
    task = get_task("qa")
    print(len(task.get_samples()))