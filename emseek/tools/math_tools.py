import math

from .base import BaseTool

class ArithmeticTool(BaseTool):
    """
    ArithmeticTool: Performs basic arithmetic operations.

    Input JSON format:
    {
        "operation": "add" | "subtract" | "multiply" | "divide",
        "operands": [<number1>, <number2>, ...]
    }

    Output JSON format on success:
    {
        "status": "success",
        "result": <computed number>,
        "message": "Operation completed successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<error description>"
    }
    """
    def __init__(self):
        description = (
            "ArithmeticTool: Performs basic arithmetic operations such as addition, subtraction, "
            "multiplication, and division.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "operation": "add" | "subtract" | "multiply" | "divide",\n'
            '    "operands": [<number1>, <number2>, ...]\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "result": <computed number>,\n'
            '    "message": "Operation completed successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<error description>"\n'
            "}"
        )
        super().__init__(name="ArithmeticTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            operation = input_json.get("operation")
            operands = input_json.get("operands")
            if operation is None or operands is None:
                raise ValueError("Missing required fields: 'operation' or 'operands'.")
            if not isinstance(operands, list) or len(operands) < 2:
                raise ValueError("'operands' must be a list containing at least two numbers.")
            
            if operation == "add":
                result = sum(operands)
            elif operation == "subtract":
                result = operands[0]
                for op in operands[1:]:
                    result -= op
            elif operation == "multiply":
                result = 1
                for op in operands:
                    result *= op
            elif operation == "divide":
                result = operands[0]
                for op in operands[1:]:
                    if op == 0:
                        raise ValueError("Division by zero is not allowed.")
                    result /= op
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return {
                "status": "success",
                "result": result,
                "message": "Operation completed successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class ExponentiationTool(BaseTool):
    """
    ExponentiationTool: Computes the power of a base raised to an exponent.

    Input JSON format:
    {
        "base": <number>,
        "exponent": <number>
    }

    Output JSON format on success:
    {
        "status": "success",
        "result": <computed number>,
        "message": "Exponentiation completed successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<error description>"
    }
    """
    def __init__(self):
        description = (
            "ExponentiationTool: Computes the result of raising a base to an exponent.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "base": <number>,\n'
            '    "exponent": <number>\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "result": <computed number>,\n'
            '    "message": "Exponentiation completed successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<error description>"\n'
            "}"
        )
        super().__init__(name="ExponentiationTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            base = input_json.get("base")
            exponent = input_json.get("exponent")
            if base is None or exponent is None:
                raise ValueError("Missing required fields: 'base' or 'exponent'.")
            
            result = math.pow(base, exponent)
            return {
                "status": "success",
                "result": result,
                "message": "Exponentiation completed successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class SquareRootTool(BaseTool):
    """
    SquareRootTool: Computes the square root of a non-negative number.

    Input JSON format:
    {
        "number": <non-negative number>
    }

    Output JSON format on success:
    {
        "status": "success",
        "result": <square root of the number>,
        "message": "Square root computed successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<error description>"
    }
    """
    def __init__(self):
        description = (
            "SquareRootTool: Computes the square root of a given non-negative number.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "number": <non-negative number>\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "result": <square root of the number>,\n'
            '    "message": "Square root computed successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<error description>"\n'
            "}"
        )
        super().__init__(name="SquareRootTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            number = input_json.get("number")
            if number is None:
                raise ValueError("Missing required field: 'number'.")
            if number < 0:
                raise ValueError("Cannot compute square root of a negative number.")
            
            result = math.sqrt(number)
            return {
                "status": "success",
                "result": result,
                "message": "Square root computed successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }