import uuid


def generate_filename(prefix="data", extension=".txt"):
    unique_id = uuid.uuid4()
    return f"{prefix}_{unique_id}{extension}"