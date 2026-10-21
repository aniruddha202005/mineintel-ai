"""Data access for query history."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import QueryLog


class QueryLogRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, log: QueryLog) -> QueryLog:
        self.db.add(log)
        self.db.flush()
        return log

    def recent(self, limit: int = 20) -> list[QueryLog]:
        statement = select(QueryLog).order_by(QueryLog.created_at.desc(), QueryLog.id.desc()).limit(limit)
        return list(self.db.scalars(statement))
