from datetime import datetime, timedelta
import uuid
from typing import Any, Dict, List, Optional, Tuple
from database import supabase

_in_memory_reviews: List[Dict[str, Any]] = []

class ReviewService:
    """Service implementing the SuperMemo SM-2 Spaced Repetition algorithm."""

    def calculate_next_review(
        self, ease_factor: float, interval: int, quality: int, repetitions: int = 1
    ) -> Tuple[float, int, int, datetime]:
        """
        Calculates updated SM-2 parameters and next review date.

        SM-2 Rules:
        - Quality: integer from 0 to 5.
        - EF' = EF + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        - EF minimum boundary is 1.3.
        - If quality < 3:
            repetitions = 0
            interval = 1
        - If quality >= 3:
            repetitions += 1
            if repetitions == 1:
                interval = 1
            elif repetitions == 2:
                interval = 6
            else:
                interval = int(round(interval * ease_factor))
        """
        quality = max(0, min(5, quality))

        # Calculate new Ease Factor
        new_ef = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        if new_ef < 1.3:
            new_ef = 1.3
        new_ef = round(new_ef, 2)

        if quality < 3:
            new_reps = 0
            new_interval = 1
        else:
            new_reps = repetitions + 1
            if new_reps == 1:
                new_interval = 1
            elif new_reps == 2:
                new_interval = 6
            else:
                new_interval = max(1, int(round(interval * new_ef)))

        next_date = datetime.utcnow() + timedelta(days=new_interval)
        return new_ef, new_interval, new_reps, next_date

    async def schedule_review(self, user_id: str, question_id: str, quality: int) -> Dict[str, Any]:
        """Schedules or updates a review item for a given user and question."""
        ef, interval, reps, next_date = self.calculate_next_review(
            ease_factor=2.5, interval=1, quality=quality, repetitions=0
        )

        record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "question_id": question_id,
            "last_reviewed": datetime.utcnow().isoformat(),
            "ease_factor": ef,
            "interval": interval,
            "repetitions": reps,
            "next_review_date": next_date.isoformat(),
        }

        _in_memory_reviews.append(record)

        try:
            res = supabase.table("review_items").insert(record).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass

        return record

    async def get_due_reviews_today(self, user_id: str) -> List[Dict[str, Any]]:
        """Returns all review items due on or before today for the specified user."""
        now_iso = datetime.utcnow().isoformat()
        results = []

        try:
            res = supabase.table("review_items").select("*").eq("user_id", user_id).lte("next_review_date", now_iso).execute()
            if res.data:
                results.extend(res.data)
        except Exception:
            pass

        if not results:
            now_dt = datetime.utcnow()
            for r in _in_memory_reviews:
                if r.get("user_id") == user_id:
                    next_dt = datetime.fromisoformat(r["next_review_date"]) if isinstance(r["next_review_date"], str) else r["next_review_date"]
                    if next_dt <= now_dt:
                        results.append(r)

        return results

    async def submit_review_item(self, user_id: str, review_item_id: str, quality: int) -> Dict[str, Any]:
        """Updates SM-2 parameters for an existing review item based on quality score (0-5)."""
        target_item = None

        try:
            res = supabase.table("review_items").select("*").eq("id", review_item_id).eq("user_id", user_id).execute()
            if res.data:
                target_item = res.data[0]
        except Exception:
            pass

        if not target_item:
            for r in _in_memory_reviews:
                if r.get("id") == review_item_id and r.get("user_id") == user_id:
                    target_item = r
                    break

        if not target_item:
            # Fallback mock creation if item ID not present
            target_item = {
                "id": review_item_id,
                "user_id": user_id,
                "question_id": str(uuid.uuid4()),
                "ease_factor": 2.5,
                "interval": 1,
                "repetitions": 1,
            }

        curr_ef = target_item.get("ease_factor", 2.5)
        curr_interval = target_item.get("interval", 1)
        curr_reps = target_item.get("repetitions", 1)

        new_ef, new_interval, new_reps, next_date = self.calculate_next_review(
            ease_factor=curr_ef, interval=curr_interval, quality=quality, repetitions=curr_reps
        )

        updates = {
            "last_reviewed": datetime.utcnow().isoformat(),
            "ease_factor": new_ef,
            "interval": new_interval,
            "repetitions": new_reps,
            "next_review_date": next_date.isoformat(),
        }

        target_item.update(updates)

        try:
            supabase.table("review_items").update(updates).eq("id", review_item_id).eq("user_id", user_id).execute()
        except Exception:
            pass

        return target_item

    async def get_upcoming_reviews(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Returns upcoming reviews for the specified user grouped by day over the next N days."""
        now = datetime.utcnow()
        end_date = now + timedelta(days=days)

        results = []
        try:
            res = supabase.table("review_items").select("*").eq("user_id", user_id).gte("next_review_date", now.isoformat()).lte("next_review_date", end_date.isoformat()).execute()
            if res.data:
                results.extend(res.data)
        except Exception:
            pass

        if not results:
            for r in _in_memory_reviews:
                if r.get("user_id") == user_id:
                    next_dt = datetime.fromisoformat(r["next_review_date"]) if isinstance(r["next_review_date"], str) else r["next_review_date"]
                    if now <= next_dt <= end_date:
                        results.append(r)

        # Group by day offset (0 to days-1)
        daily_counts = { (now + timedelta(days=i)).strftime("%Y-%m-%d"): 0 for i in range(days) }
        for r in results:
            next_dt = datetime.fromisoformat(r["next_review_date"]) if isinstance(r["next_review_date"], str) else r["next_review_date"]
            day_str = next_dt.strftime("%Y-%m-%d")
            if day_str in daily_counts:
                daily_counts[day_str] += 1

        return {
            "user_id": user_id,
            "total_upcoming": len(results),
            "days_ahead": days,
            "schedule": daily_counts,
            "items": results
        }

review_service = ReviewService()
