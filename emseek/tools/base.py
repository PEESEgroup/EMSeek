from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    A generic tool class that provides a template for various tools.
    
    Each tool has:
      - a name
      - a basic description
      - an execute() method to perform its primary function.
      
    Subclasses should override the execute() method to implement specific functionalities.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_description(self) -> str:
        """
        Return the basic description of the tool.
        """
        return self.description

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the tool's primary function.
        This method should be implemented by subclasses.
        """
        pass