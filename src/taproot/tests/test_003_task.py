# Tests for the task module
import time
import asyncio

from taproot import Task, TaskQueue
from taproot.util import debug_logger
from taproot.constants import NOTSET

class TaskTestSquare(Task):
    """
    A simple task that squares a number.
    """
    task = "task_test_square"
    default = True

    def __call__(self, *, x: float) -> float: # type: ignore[override]
        return x * x

class StepTaskTestSquare(Task):
    """
    A simple task that squares a number.
    """
    task = "task_test_square"
    model = "steps"

    def __call__( # type: ignore[override]
        self,
        *,
        x: float,
        num_steps: int = 10
    ) -> float:
        """
        Squares a number after a delay.
        """
        self.num_steps = num_steps
        for step in self.step_iterator:
            time.sleep(0.02)
            self.add_intermediate(step)
        return x * x

def test_task() -> None:
    with debug_logger() as logger:
        catalog = Task.catalog()
        assert "task_test_square" in catalog
        assert catalog["task_test_square"]["default"] is None
        assert catalog["task_test_square"]["models"][None]["task"] is TaskTestSquare
        assert "x" in catalog["task_test_square"]["models"][None]["parameters"]
        assert catalog["task_test_square"]["models"][None]["parameters"]["x"]["parameter_type"] is float
        assert catalog["task_test_square"]["models"][None]["parameters"]["x"]["default"] is NOTSET
        assert catalog["task_test_square"]["models"][None]["parameters"]["x"]["required"]
        assert catalog["task_test_square"]["models"][None]["return_type"] is float
        assert catalog["task_test_square"]["models"][None]["task"]()(x=3) == 9

def test_task_queue() -> None:
    with debug_logger() as logger:
        # Create a task queue
        task_queue = TaskQueue({"task": "task_test_square"})

        # Arguments for the task
        queue_kwargs = {"x": 3}

        # Queue the task
        queue_result = task_queue(**queue_kwargs)
        assert queue_result["status"] == "new"
        # Store the ID
        queue_id = queue_result["id"]

        # Sleep for a bit
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(0.1))

        # By now it should be complete
        queue_result_complete = task_queue(**queue_kwargs)
        assert queue_result_complete["id"] == queue_id # Should be the same ID
        assert queue_result_complete["status"] == "complete"
        assert queue_result_complete["result"] == 9

        # Queue with the same arguments and get an immediate result
        queue_result_cached = task_queue(**queue_kwargs)
        assert queue_result_cached["id"] == queue_id
        assert queue_result_cached["status"] == "complete"
        assert queue_result_cached["result"] == 9

        # Now queue with an IO-blocking task
        delayed_echo_queue = TaskQueue({"task": "echo"})

        # Queue a task
        delayed_echo_result = delayed_echo_queue(
            message="Hello, world!",
            delay=1.0
        )
        assert delayed_echo_result["status"] == "new"

        # Save the ID
        delayed_echo_id = delayed_echo_result["id"]

        # Sleep for a bit
        loop.run_until_complete(asyncio.sleep(0.2))

        # By now it should be active, but not complete
        # Get it using its ID only
        delayed_echo_result_active = delayed_echo_queue(id=delayed_echo_id)
        assert delayed_echo_result_active["status"] == "active"
        assert delayed_echo_result_active["result"] is None

        # Sleep for a bit
        loop.run_until_complete(asyncio.sleep(0.9))

        # By now it should be complete
        delayed_echo_result_complete = delayed_echo_queue(id=delayed_echo_id)
        assert delayed_echo_result_complete["status"] == "complete"
        assert delayed_echo_result_complete["result"] == "Hello, world!"

        # Run it again and get the cached result
        delayed_echo_result_cached = delayed_echo_queue(id=delayed_echo_id)
        assert delayed_echo_result_cached["status"] == "complete"
        assert delayed_echo_result_cached["result"] == "Hello, world!"

        # Assert the activity has been tracked
        assert delayed_echo_queue.activity > 0.4
        assert task_queue.activity < 0.4

def test_task_with_steps() -> None:
    with debug_logger() as logger:
        # Create a task queue
        task_queue = TaskQueue({"task": "task_test_square", "model": "steps"})

        # Arguments for the task
        queue_kwargs = {"x": 4, "num_steps": 10}

        # Queue the task
        queue_result = task_queue(**queue_kwargs)
        assert queue_result["status"] == "new"
        # Store the ID
        queue_id = queue_result["id"]

        # Sleep for a bit, this will run for ~0.2 seconds
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(0.1))

        # By now it should be going
        queue_result_active = task_queue(**queue_kwargs)
        assert queue_result_active["id"] == queue_id # Should be the same ID
        assert queue_result_active["status"] == "active"
        assert 0.0 < queue_result_active["progress"] <= 0.5
        assert isinstance(queue_result_active["rate"], float)
        assert 2.5 < queue_result_active["rate"] < 5.5 # Theoretical rate is 5.0
        assert queue_result_active["result"] is None
        assert isinstance(queue_result_active["intermediate"], int)
        assert queue_result_active["intermediate"] >= 1

        # Wait for the result
        loop.run_until_complete(asyncio.sleep(0.2))
        queue_result_complete = task_queue(**queue_kwargs)
        assert queue_result_complete["id"] == queue_id # Should be the same ID
        assert queue_result_complete["status"] == "complete"
        assert queue_result_complete["result"] == 16
        assert queue_result_complete["progress"] == 1.0
        assert queue_result_complete["intermediate"] is None
        assert isinstance(queue_result_complete["rate"], float)
        assert 2.5 < queue_result_complete["rate"] < 5.5 # Theoretical rate is 5.0
