"""
test_thread_building.py — Unit tests for prepare_data.py and build_threads.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prepare_data import normalize, deduplicate_text, build_conversation_index
import pandas as pd


class TestNormalize:
    def test_removes_urls(self):
        text = "check out https://example.com for help"
        result = normalize(text)
        assert "http" not in result
        assert "[url]" in result

    def test_removes_mentions(self):
        text = "@SpotifyCares I can't login"
        result = normalize(text)
        assert "@spotifycares" not in result
        assert "login" in result

    def test_lowercases(self):
        result = normalize("THIS IS LOUD")
        assert result == "this is loud"

    def test_collapses_whitespace(self):
        result = normalize("  too   many   spaces  ")
        assert "  " not in result

    def test_keeps_mentions_if_flag_set(self):
        text = "@SpotifyCares hello"
        result = normalize(text, keep_mentions=True)
        assert "@spotifycares" in result


class TestDeduplication:
    def test_keeps_unique(self):
        texts = ["I cannot login", "songs keep pausing", "app keeps crashing"]
        kept = deduplicate_text(texts, threshold=0.8)
        assert len(kept) == 3

    def test_removes_near_duplicate(self):
        texts = [
            "spotify songs keep pausing all the time",
            "spotify songs keep pausing all the time today",  # near-dup
            "app keeps crashing on iphone",
        ]
        kept = deduplicate_text(texts, threshold=0.8)
        assert len(kept) == 2  # one duplicate removed

    def test_empty_list(self):
        kept = deduplicate_text([])
        assert kept == []

    def test_single_item(self):
        kept = deduplicate_text(["hello"])
        assert kept == [0]


class TestConversationIndex:
    def test_basic_chain(self):
        """Two tweets: customer → brand reply."""
        df = pd.DataFrame([
            {
                "tweet_id": "t1",
                "author_id": "customer1",
                "inbound": True,
                "created_at": "2017-01-01",
                "text": "help!",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "t2",
                "author_id": "spotifycares",
                "inbound": False,
                "created_at": "2017-01-01",
                "text": "Hi! We can help.",
                "in_response_to_tweet_id": "t1",
            },
        ])
        threads = build_conversation_index(df)
        # t1 should be the root, thread should contain both
        assert "t1" in threads
        assert "t2" in threads["t1"]

    def test_no_responses(self):
        """Single tweet with no replies."""
        df = pd.DataFrame([
            {
                "tweet_id": "t1",
                "author_id": "customer1",
                "inbound": True,
                "created_at": "2017-01-01",
                "text": "help!",
                "in_response_to_tweet_id": None,
            }
        ])
        threads = build_conversation_index(df)
        assert "t1" in threads
        assert threads["t1"] == ["t1"]

    def test_multi_level_chain(self):
        """Three tweets in sequence: customer → brand → customer."""
        df = pd.DataFrame([
            {"tweet_id": "t1", "author_id": "c1", "inbound": True, "created_at": "2017-01-01", "text": "help", "in_response_to_tweet_id": None},
            {"tweet_id": "t2", "author_id": "spotifycares", "inbound": False, "created_at": "2017-01-01", "text": "how?", "in_response_to_tweet_id": "t1"},
            {"tweet_id": "t3", "author_id": "c1", "inbound": True, "created_at": "2017-01-01", "text": "login error", "in_response_to_tweet_id": "t2"},
        ])
        threads = build_conversation_index(df)
        assert "t1" in threads
        assert threads["t1"] == ["t1", "t2", "t3"]

    def test_missing_parent(self):
        """Tweet pointing to a parent that is not in the dataset should act as root."""
        df = pd.DataFrame([
            {"tweet_id": "t2", "author_id": "spotifycares", "inbound": False, "created_at": "2017-01-01", "text": "replying to missing", "in_response_to_tweet_id": "t_missing"},
        ])
        threads = build_conversation_index(df)
        assert "t2" in threads
        assert threads["t2"] == ["t2"]

