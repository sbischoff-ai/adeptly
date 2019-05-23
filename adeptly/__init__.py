"""Spawn adaptive agents in real-time environments."""

import tensorflow

from adeptly.dqn import DQNAgent


class AdeptlyEngine:
    """Handle Tensorflow standard graph reference.

    Usage:
    ```python
    from adeptly import AdeptlyEngine

    with AdeptlyEngine():
        # Do stuff with adeptly agents.
    ```

    """

    graph = tensorflow.get_default_graph()

    def __new__(cls):
        return cls.graph.as_default()
