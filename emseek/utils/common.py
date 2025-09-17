import json


def stream_json(data):
    """
    Converts the given JSON-serializable object (list or dictionary) into a streaming output.
    
    If the input is a list, it streams each element as part of a JSON array.
    If the input is a dictionary, it streams each key-value pair as part of a JSON object.
    Otherwise, it serializes the object into a complete JSON string.
    
    Returns:
        A generator that yields parts of a JSON-formatted string.
    """
    if isinstance(data, list):
        yield '['  # Yield the beginning of a JSON array
        first = True
        for item in data:
            if not first:
                yield ','  # Insert a comma separator for elements after the first one
            else:
                first = False
            # Serialize the current item into a JSON string
            yield json.dumps(item)
        yield ']'  # Yield the end of the JSON array
    elif isinstance(data, dict):
        yield '{'  # Yield the beginning of a JSON object
        first = True
        for key, value in data.items():
            if not first:
                yield ','  # Insert a comma separator for key-value pairs after the first one
            else:
                first = False
            # Serialize the key and value, ensuring keys are stringified in JSON
            yield json.dumps(key) + ':' + json.dumps(value)
        yield '}'  # Yield the end of the JSON object
    else:
        # For other data types, serialize the entire object at once
        yield json.dumps(data)