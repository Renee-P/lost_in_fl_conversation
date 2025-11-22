from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import json

class Task(ABC):
    """Base class for all tasks"""

    def __init__(self, version: str):
        self.version: str = version
        self.task_name: str = self._get_task_name()
        self.answer_description: str = self._get_answer_description()
        self.answer_extraction_strategy: str = self._get_answer_extraction_strategy() # this can be "prefix_suffix", "gen", "full_response"

    @abstractmethod
    def get_task_name(self) -> str:
        """Return the name of the task (e.g., 'humaneval', 'spider')"""

    @abstractmethod
    def get_dataset_file(self) -> str:
        """Return the name of the file containing the samples for the task"""
        pass

    @abstractmethod
    def get_samples(self, version: str, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a list of samples for the task"""
        pass

    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        samples = self.get_samples()
        id2samples = {sample["task_id"]: sample for sample in samples}
        if sample_id not in id2samples:
            raise ValueError(f"Sample ID {sample_id} not found")
        return id2samples[sample_id]

    @abstractmethod
    def get_answer_description(self) -> str:
        """Return description of what constitutes a valid answer"""
        pass

    @abstractmethod
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the system prompt for the given sample"""
        pass

    @abstractmethod
    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> Union[bool, tuple[bool, str]]:
        """Evaluate if the extracted answer is correct.
        Returns either a boolean or a tuple of (boolean, feedback_string)"""
        pass

    @abstractmethod
    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the populated prompt for fully-specified attempts"""
        pass

    @abstractmethod
    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the populated prompt for concatenated experiment"""
        pass

    @abstractmethod
    def populate_full_examples(self, num_examples: int = 5) -> str:
        """Generate few-shot examples for full prompts."""
        pass
    
    @abstractmethod
    def populate_sharded_examples(self, num_examples: int = 5) -> str:
        """Generate few-shot examples for sharded/concat prompts."""
        pass

    def save_samples(self, samples: List[Dict[str, Any]]):
        dataset_fn = self.get_dataset_file()
        with open(dataset_fn, "w") as f:
            json.dump(samples, f, indent=4)

    @abstractmethod
    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Given a sample in the dataset file, return a dictionary with all the information from the original sample; helpful for displaying the sample in the annotation UI"""
        pass