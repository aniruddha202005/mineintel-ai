"""Domain exceptions that map directly onto HTTP status codes."""


class MineIntelError(Exception):
    """Base class for errors the API reports to the client."""

    status_code = 500


class InvalidDocumentError(MineIntelError):
    status_code = 422


class FileTooLargeError(MineIntelError):
    status_code = 413


class EmptyDocumentError(MineIntelError):
    status_code = 422


class DuplicateDocumentError(MineIntelError):
    status_code = 409


class DocumentNotFoundError(MineIntelError):
    status_code = 404


class LLMError(MineIntelError):
    status_code = 502


class EmbeddingError(MineIntelError):
    status_code = 502
