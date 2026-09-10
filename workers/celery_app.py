from celery import Celery
from celery.schedules import crontab
from config import settings

celery_app = Celery(
    "focusly_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "daily-review-queue-cron": {
            "task": "workers.tasks.daily_review_queue_task",
            "schedule": crontab(hour=0, minute=0), # Run daily at midnight UTC
        },
        "monthly-usage-reset-cron": {
            "task": "workers.tasks.reset_monthly_usage_task",
            "schedule": crontab(0, 0, day_of_month="1"), # Run on the 1st of every month at 00:00 UTC
        },
    },
)
