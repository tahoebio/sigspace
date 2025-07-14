"""Base task class for Tahoe Agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTask(ABC):
    """Base class for all tasks in Tahoe Agent."""

    def __init__(
        self,
        name: str,
        description: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        requirements: Optional[List[str]] = None,
    ):
        """Initialize the base task.

        Args:
            name: Name of the task
            description: Description of what the task does
            inputs: Dictionary describing the task's inputs
            outputs: Dictionary describing the task's outputs
            requirements: List of requirements or dependencies
        """
        self.name = name
        self.description = description
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.requirements = requirements or []
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.error = None

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the task with the given parameters.

        Args:
            **kwargs: Parameters for task execution

        Returns:
            Result of task execution
        """
        pass

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that all required inputs are provided.

        Args:
            **kwargs: Inputs to validate

        Returns:
            True if all required inputs are present

        Raises:
            ValueError: If required inputs are missing
        """
        required_inputs = [
            key for key, spec in self.inputs.items() if spec.get("required", False)
        ]
        missing_inputs = [inp for inp in required_inputs if inp not in kwargs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")
        return True

    def run(self, **kwargs) -> Any:
        """Run the task with input validation and error handling.

        Args:
            **kwargs: Parameters for task execution

        Returns:
            Result of task execution
        """
        try:
            self.status = "running"
            self.validate_inputs(**kwargs)
            self.result = self.execute(**kwargs)
            self.status = "completed"
            return self.result
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            raise

    def to_schema(self) -> Dict[str, Any]:
        """Convert the task to a schema dictionary.

        Returns:
            Dictionary representation of the task
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "requirements": self.requirements,
            "status": self.status,
        }

    def reset(self):
        """Reset the task status and clear results."""
        self.status = "pending"
        self.result = None
        self.error = None

    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task(name='{self.name}', status='{self.status}')"

    def __repr__(self) -> str:
        """Detailed string representation of the task."""
        return (
            f"Task(name='{self.name}', description='{self.description}', "
            f"status='{self.status}', inputs={self.inputs}, outputs={self.outputs})"
        )


class SimpleTask(BaseTask):
    """A simple task that executes a provided function."""

    def __init__(
        self,
        name: str,
        description: str,
        func: callable,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the simple task.

        Args:
            name: Name of the task
            description: Description of the task
            func: Function to execute
            inputs: Input specification
            outputs: Output specification
        """
        super().__init__(name, description, inputs, outputs)
        self.func = func

    def execute(self, **kwargs) -> Any:
        """Execute the function with the given parameters.

        Args:
            **kwargs: Parameters for function execution

        Returns:
            Result of function execution
        """
        return self.func(**kwargs)


class DataAnalysisTask(BaseTask):
    """A task specifically designed for data analysis."""

    def __init__(
        self,
        name: str,
        description: str,
        data_source: str,
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the data analysis task.

        Args:
            name: Name of the task
            description: Description of the task
            data_source: Source of the data to analyze
            analysis_type: Type of analysis to perform
            parameters: Analysis parameters
        """
        inputs = {
            "data_source": {
                "type": "string",
                "required": True,
                "description": "Path to data source",
            },
            "parameters": {
                "type": "object",
                "required": False,
                "description": "Analysis parameters",
            },
        }
        outputs = {
            "result": {"type": "object", "description": "Analysis results"},
            "summary": {"type": "string", "description": "Summary of analysis"},
        }

        super().__init__(name, description, inputs, outputs)
        self.data_source = data_source
        self.analysis_type = analysis_type
        self.parameters = parameters or {}

    def execute(
        self,
        data_source: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the data analysis.

        Args:
            data_source: Path to data source (overrides default if provided)
            parameters: Analysis parameters (overrides default if provided)

        Returns:
            Dictionary containing analysis results and summary
        """
        # Use provided parameters or defaults
        source = data_source or self.data_source
        params = parameters or self.parameters

        # This is a placeholder implementation
        # In a real implementation, you would load data and perform actual analysis
        return {
            "result": {
                "data_source": source,
                "analysis_type": self.analysis_type,
                "parameters": params,
                "status": "completed",
            },
            "summary": f"Completed {self.analysis_type} analysis on {source}",
        }


class WorkflowTask(BaseTask):
    """A task that combines multiple sub-tasks in sequence."""

    def __init__(
        self,
        name: str,
        description: str,
        subtasks: List[BaseTask],
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the workflow task.

        Args:
            name: Name of the task
            description: Description of the task
            subtasks: List of subtasks to execute in order
            inputs: Input specification
            outputs: Output specification
        """
        super().__init__(name, description, inputs, outputs)
        self.subtasks = subtasks

    def execute(self, **kwargs) -> List[Any]:
        """Execute all subtasks in sequence.

        Args:
            **kwargs: Parameters for task execution

        Returns:
            List of results from each subtask
        """
        results = []
        for subtask in self.subtasks:
            try:
                result = subtask.run(**kwargs)
                results.append(result)
                # Pass results forward to next task if needed
                if hasattr(subtask, "result") and subtask.result:
                    kwargs.update({"previous_result": subtask.result})
            except Exception as e:
                # Handle subtask failure
                results.append({"error": str(e), "task": subtask.name})
                # You might want to continue or stop based on requirements
                break

        return results

    def get_subtask_status(self) -> Dict[str, str]:
        """Get the status of all subtasks.

        Returns:
            Dictionary mapping subtask names to their status
        """
        return {task.name: task.status for task in self.subtasks}

    def reset(self):
        """Reset the workflow and all subtasks."""
        super().reset()
        for subtask in self.subtasks:
            subtask.reset()
