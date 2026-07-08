"""Tests for the pluggable usage-persistence layer."""

import threading

import pytest

from selfllm.serving.billing import BillingManager, Tier, DEFAULT_TIERS
from selfllm.serving.usage_store import (
    InMemoryUsageStore,
    SQLiteUsageStore,
    usage_store_from_env,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryUsageStore()
    return SQLiteUsageStore(str(tmp_path / "usage.db"))


class TestUsageStoreContract:
    """Behavior shared by every UsageStore backend."""

    def test_get_unseen_is_zero(self, store):
        assert store.get("sk-a", "2026-06-28") == (0, 0, 0)

    def test_add_accumulates_and_returns_totals(self, store):
        assert store.add("sk-a", "2026-06-28", 1, 10, 5) == (1, 10, 5)
        assert store.add("sk-a", "2026-06-28", 1, 3, 2) == (2, 13, 7)
        assert store.get("sk-a", "2026-06-28") == (2, 13, 7)

    def test_key_and_day_are_isolated(self, store):
        store.add("sk-a", "2026-06-28", 1, 1, 1)
        store.add("sk-b", "2026-06-28", 1, 2, 2)
        store.add("sk-a", "2026-06-29", 1, 4, 4)
        assert store.get("sk-a", "2026-06-28") == (1, 1, 1)
        assert store.get("sk-b", "2026-06-28") == (1, 2, 2)
        assert store.get("sk-a", "2026-06-29") == (1, 4, 4)

    def test_concurrent_adds_are_atomic(self, store):
        def worker():
            for _ in range(50):
                store.add("sk-a", "2026-06-28", 1, 1, 0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        requests, prompt_tokens, _ = store.get("sk-a", "2026-06-28")
        assert requests == 8 * 50
        assert prompt_tokens == 8 * 50


class TestSQLitePersistence:
    def test_survives_reopen(self, tmp_path):
        path = str(tmp_path / "usage.db")
        store = SQLiteUsageStore(path)
        store.add("sk-a", "2026-06-28", 3, 100, 40)
        store.close()

        # A fresh store on the same file sees the persisted counters -- this
        # is the whole point: quotas survive a process restart.
        reopened = SQLiteUsageStore(path)
        assert reopened.get("sk-a", "2026-06-28") == (3, 100, 40)

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "usage.db"
        store = SQLiteUsageStore(str(nested))
        store.add("k", "d", 1, 0, 0)
        assert nested.exists()


class TestUsageStoreFromEnv:
    def test_env_unset_is_in_memory(self):
        assert isinstance(usage_store_from_env({}), InMemoryUsageStore)

    def test_env_path_is_sqlite(self, tmp_path):
        store = usage_store_from_env(
            {"SELFLLM_USAGE_DB": str(tmp_path / "u.db")}
        )
        assert isinstance(store, SQLiteUsageStore)


class TestBillingPersistence:
    """The BillingManager honors quotas across a simulated restart."""

    def test_quota_persists_across_manager_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "selfllm.serving.billing._today", lambda: "2026-06-28"
        )
        path = str(tmp_path / "usage.db")
        tiers = dict(DEFAULT_TIERS)
        tiers["tiny"] = Tier("tiny", requests_per_day=3, tokens_per_day=None,
                             price_per_month_usd=0.0)

        mgr = BillingManager(keys={"sk-a": "tiny"}, tiers=tiers,
                             store=SQLiteUsageStore(path))
        mgr.record("sk-a", 5, 5)
        mgr.record("sk-a", 5, 5)
        mgr._store.close()

        # New manager (as if the process restarted) on the same DB: the two
        # prior requests still count, so a third is the last allowed.
        mgr2 = BillingManager(keys={"sk-a": "tiny"}, tiers=tiers,
                              store=SQLiteUsageStore(path))
        assert mgr2.usage_report("sk-a")["requests"] == 2
        mgr2.check_quota("sk-a")          # 2 < 3, still ok
        mgr2.record("sk-a", 1, 1)         # now 3
        with pytest.raises(Exception):    # QuotaExceeded
            mgr2.check_quota("sk-a")

    def test_from_env_uses_sqlite_when_configured(self, tmp_path):
        mgr = BillingManager.from_env(
            {"SELFLLM_API_KEYS": "sk-a:free",
             "SELFLLM_USAGE_DB": str(tmp_path / "u.db")}
        )
        assert isinstance(mgr._store, SQLiteUsageStore)
        mgr.record("sk-a", 1, 1)
        assert mgr.usage_report("sk-a")["total_tokens"] == 2
