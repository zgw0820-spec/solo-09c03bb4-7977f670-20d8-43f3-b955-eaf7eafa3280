from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import EngineConfig, LinkConfig
from app.engine import DrillEngine
from app.main import create_app


def make_client(a_delivery=60, a_ack=30, b_delivery=30, b_ack=15) -> TestClient:
    config = EngineConfig(
        link_a=LinkConfig("A", a_delivery, a_ack),
        link_b=LinkConfig("B", b_delivery, b_ack),
    )
    return TestClient(create_app(config))


def wait_for(client: TestClient, predicate, timeout_s: float = 8.0, interval_s: float = 0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def removal(client: TestClient, removal_id: str) -> dict:
    response = client.get(f"/api/removals/{removal_id}")
    assert response.status_code == 200, response.text
    return response.json()


def received(client: TestClient, flow_id: str) -> list[dict]:
    response = client.get(f"/api/flows/{flow_id}/received")
    assert response.status_code == 200, response.text
    return response.json()["delivery_order"]


def create_flow(client: TestClient, flow_id: str, link: str) -> dict:
    response = client.post("/api/flows", json={"flow_id": flow_id, "link": link})
    assert response.status_code == 201, response.text
    return response.json()


def send(client: TestClient, flow_id: str, seq: int, payload=None, expect=201):
    response = client.post(
        f"/api/flows/{flow_id}/packets", json={"seq": seq, "payload": payload}
    )
    assert response.status_code == expect, response.text
    return response.json()


@pytest.fixture()
def client():
    with make_client() as test_client:
        yield test_client


def test_health_and_link_defaults(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    links = {link["id"]: link for link in client.get("/api/links").json()["links"]}
    assert links["A"]["delivery_delay_ms"] == 60
    assert links["B"]["ack_delay_ms"] == 15


def test_fifo_in_order_delivery_and_duplicate_rejection(client):
    create_flow(client, "f1", "A")
    for seq in range(1, 6):
        send(client, "f1", seq)
    # Duplicate seq is not accepted; no extra copy enters the pipe.
    send(client, "f1", 3, expect=409)

    assert wait_for(client, lambda: len(received(client, "f1")) == 5)
    order = [item["seq"] for item in received(client, "f1")]
    assert order == [1, 2, 3, 4, 5]
    assert all(item["link"] == "A" for item in received(client, "f1"))
    assert [item["delivery_index"] for item in received(client, "f1")] == [1, 2, 3, 4, 5]

    flow = client.get("/api/flows/f1").json()
    assert flow["accepted_count"] == 5
    assert flow["count"] == 5


def test_removal_success_moves_flows_and_buffered_packets(client):
    create_flow(client, "f-a", "A")
    create_flow(client, "f-b", "B")
    # Two packets already queued on the member link must be confirmed first.
    send(client, "f-a", 1)
    send(client, "f-a", 2)

    response = client.post(
        "/api/removals",
        json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 5},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["snapshot_packet_count"] == 2
    removal_id = body["removal_id"]

    # Second concurrent request is refused.
    busy = client.post(
        "/api/removals",
        json={"member_link": "B", "wait_ms": 1000, "buffer_limit": 1},
    )
    assert busy.status_code == 409

    # New packets for the affected flow are buffered (202), other flows flow on.
    buffered = send(client, "f-a", 3, expect=202)
    assert buffered["buffered"] is True
    send(client, "f-b", 1)

    late = client.post(
        "/api/removals",
        json={"member_link": "A", "wait_ms": 1000, "buffer_limit": 1},
    )
    assert late.status_code == 409  # still pending

    assert wait_for(
        client, lambda: removal(client, removal_id)["status"] == "succeeded"
    )
    detail = removal(client, removal_id)
    assert detail["result"] == "succeeded"
    assert detail["reason"] == "peer_confirmed"
    assert detail["snapshot_acked"] == 2
    assert detail["moved_flow_ids"] == ["f-a"]
    assert len(detail["flushed_packet_ids"]) == 1
    assert detail["flushed_to"] == "B"

    assert wait_for(client, lambda: len(received(client, "f-a")) == 3)
    order = received(client, "f-a")
    assert [item["seq"] for item in order] == [1, 2, 3]
    assert [item["link"] for item in order] == ["A", "A", "B"]

    assert wait_for(client, lambda: len(received(client, "f-b")) == 1)
    assert received(client, "f-b")[0]["link"] == "B"


def test_buffer_limit_rejects_but_other_flows_continue(client):
    create_flow(client, "f-a", "A")
    create_flow(client, "f-other", "B")
    send(client, "f-a", 1)  # keeps the removal pending for ~90ms

    response = client.post(
        "/api/removals",
        json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 1},
    )
    removal_id = response.json()["removal_id"]

    first = send(client, "f-a", 2, expect=202)
    assert first["buffered"] is True
    rejected = client.post("/api/flows/f-a/packets", json={"seq": 3, "payload": None})
    assert rejected.status_code == 429
    assert rejected.json()["buffer_limit"] == 1
    assert rejected.json()["buffered_count"] == 1

    # Rejected seq was never accepted, so it can be retried after the removal.
    send(client, "f-other", 1)
    assert wait_for(
        client, lambda: removal(client, removal_id)["status"] == "succeeded"
    )
    detail = removal(client, removal_id)
    assert detail["rejected_count"] == 1
    # send_packet ordering: accepted packet 2 is flushed; seq 3 can now be sent
    send(client, "f-a", 3)
    assert wait_for(client, lambda: len(received(client, "f-a")) == 3)
    assert [item["seq"] for item in received(client, "f-a")] == [1, 2, 3]


def test_timeout_keeps_flows_and_flushes_buffer_to_original_link(client):
    create_flow(client, "f-a", "A")
    create_flow(client, "f-b", "B")
    send(client, "f-a", 1)  # takes 90ms on A

    response = client.post(
        "/api/removals",
        json={"member_link": "A", "wait_ms": 25, "buffer_limit": 4},
    )
    removal_id = response.json()["removal_id"]
    send(client, "f-a", 2, expect=202)
    send(client, "f-a", 3, expect=202)

    assert wait_for(
        client, lambda: removal(client, removal_id)["status"] == "timed_out"
    )
    detail = removal(client, removal_id)
    assert detail["result"] == "timed_out"
    assert detail["moved_flow_ids"] == []
    assert len(detail["flushed_packet_ids"]) == 2
    assert detail["flushed_to"] == "A"

    # Late acknowledgement arrives after the timeout and must not change it.
    assert wait_for(client, lambda: len(received(client, "f-a")) == 3)
    detail = removal(client, removal_id)
    assert detail["status"] == "timed_out"
    assert detail["late_confirmations_ignored"] >= 1
    assert detail["moved_flow_ids"] == []

    flow = client.get("/api/flows/f-a").json()
    assert flow["link"] == "A"
    order = received(client, "f-a")
    assert [item["seq"] for item in order] == [1, 2, 3]
    assert [item["link"] for item in order] == ["A", "A", "A"]

    # Duplicate seq after timeout is still rejected: the buffered copy was kept.
    send(client, "f-a", 2, expect=409)

    # A new removal after the timeout is allowed. Whether it is confirmed
    # immediately or has to wait depends on what is still queued on A.
    new = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 1}
    )
    assert new.status_code in (200, 202), new.text
    new_id = new.json()["removal_id"]
    assert wait_for(
        client, lambda: removal(client, new_id)["status"] == "succeeded"
    )
    assert client.get("/api/flows/f-a").json()["link"] == "B"


def test_send_to_unknown_flow_and_bad_payloads(client):
    assert client.post("/api/flows/x/packets", json={"seq": 1}).status_code == 404
    assert (
        client.post("/api/flows", json={"flow_id": "f", "link": "C"}).status_code == 422
    )
    create_flow(client, "f", "A")
    assert client.post("/api/flows", json={"flow_id": "f", "link": "A"}).status_code == 409
    assert client.post("/api/flows/f/packets", json={"seq": 0}).status_code == 422
    assert client.patch("/api/links/Z", json={"ack_delay_ms": 5}).status_code == 404


def test_link_delay_update_takes_effect_for_new_packets(client):
    assert client.patch(
        "/api/links/A", json={"delivery_delay_ms": 5, "ack_delay_ms": 5}
    ).status_code == 200
    create_flow(client, "f", "A")
    started = time.monotonic()
    send(client, "f", 1)
    assert wait_for(client, lambda: len(received(client, "f")) == 1, timeout_s=1.0)
    assert time.monotonic() - started < 0.5


def test_flow_cannot_be_created_on_member_link_while_pending(client):
    create_flow(client, "f-a", "A")
    send(client, "f-a", 1)
    client.post("/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 2})
    denied = client.post("/api/flows", json={"flow_id": "new", "link": "A"})
    assert denied.status_code == 409
    allowed = client.post("/api/flows", json={"flow_id": "new-b", "link": "B"})
    assert allowed.status_code == 201


def test_successful_removal_takes_member_link_out_of_bundle(client):
    create_flow(client, "f-a", "A")
    send(client, "f-a", 1)
    response = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 2}
    )
    removal_id = response.json()["removal_id"]
    assert wait_for(
        client, lambda: removal(client, removal_id)["status"] == "succeeded"
    )
    links = {link["id"]: link for link in client.get("/api/links").json()["links"]}
    assert links["A"]["removed"] is True
    assert links["B"]["removed"] is False

    assert client.post("/api/flows", json={"flow_id": "x", "link": "A"}).status_code == 409
    again = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 1000, "buffer_limit": 2}
    )
    assert again.status_code == 409

    # Removing the surviving member leaves no target, and is refused.
    last = client.post(
        "/api/removals", json={"member_link": "B", "wait_ms": 1000, "buffer_limit": 2}
    )
    assert last.status_code == 409

    # Reset brings both links back.
    assert client.post("/api/reset").status_code == 200
    links = {link["id"]: link for link in client.get("/api/links").json()["links"]}
    assert links["A"]["removed"] is False
    assert client.post("/api/flows", json={"flow_id": "again", "link": "A"}).status_code == 201


def test_reset_clears_flows_and_removals(client):
    create_flow(client, "f", "A")
    send(client, "f", 1)
    client.post("/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 2})
    client.post("/api/reset")
    assert client.get("/api/flows").json()["flows"] == []
    assert client.get("/api/removals").json()["removals"] == []
    state = client.get("/api/state").json()
    assert state["active_removal"] is None
    assert all(link["queue_depth"] == 0 for link in state["links"])


def test_buffer_is_per_request_and_released_at_the_end(client):
    create_flow(client, "f-a", "A")
    send(client, "f-a", 1)
    removal_id = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 2}
    ).json()["removal_id"]
    send(client, "f-a", 2, expect=202)
    send(client, "f-a", 3, expect=202)
    assert wait_for(
        client, lambda: removal(client, removal_id)["status"] == "succeeded"
    )
    # Both buffered packets were released after the switch, so the buffer is
    # empty again and new packets queue normally on the new link.
    send(client, "f-a", 4)
    assert wait_for(client, lambda: len(received(client, "f-a")) == 4)
    assert [x["seq"] for x in received(client, "f-a")] == [1, 2, 3, 4]
    assert [x["link"] for x in received(client, "f-a")] == ["A", "B", "B", "B"]


def test_static_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "聚合口摘链演练" in response.text
    assert "https://" not in response.text  # no external assets needed offline


# --------------------------------------------------------------------------
# Deadline-driven timeout semantics
# --------------------------------------------------------------------------


def test_deadline_is_measured_from_request_acceptance(client):
    """The wait clock starts when the request is accepted; time already spent
    on the link and sends made during the wait do not extend the deadline."""
    client.patch("/api/links/A", json={"delivery_delay_ms": 2000, "ack_delay_ms": 0})
    create_flow(client, "f", "A")
    send(client, "f", 1)  # will not be acked for ~2s
    body = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 500, "buffer_limit": 4}
    ).json()
    assert body["status"] == "pending"
    assert body["created_at"] is not None and body["deadline_at"] is not None

    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    # deadline = acceptance time + wait_ms
    delta = parse(body["deadline_at"]) - parse(body["created_at"])
    assert 0.49 <= delta.total_seconds() <= 0.52

    # Sends during the wait do not push the deadline out.
    time.sleep(0.2)
    send(client, "f", 2, expect=202)
    detail = removal(client, body["removal_id"])
    assert detail["deadline_at"] == body["deadline_at"]
    assert detail["status"] in {"pending", "succeeded"}


def test_confirmation_at_deadline_is_late_even_before_timeout_callback():
    """A confirmation processed after the deadline must lose even when the
    timer callback never ran (simulated by cancelling it)."""

    async def run() -> None:
        engine = DrillEngine(link_a_delivery_ms=60, link_a_ack_ms=10,
                             link_b_delivery_ms=0, link_b_ack_ms=0)
        await engine.start()
        try:
            await engine.create_flow("f", "A")
            await engine.send_packet("f", 1)
            removal = await engine.create_removal("A", wait_ms=20, buffer_limit=4)
            assert removal["status"] == "pending"
            rm = engine.removals[removal["removal_id"]]
            assert rm.timeout_task is not None
            # Starve the timeout callback: the ack (at ~70ms) will arrive well
            # after the 20ms deadline with no timer left to finalize it.
            rm.timeout_task.cancel()
            rm.timeout_task = None
            await asyncio.sleep(0.15)

            result = await engine.get_removal(removal["removal_id"])
            assert result["status"] == "timed_out", result
            assert result["late_confirmations_ignored"] == 1
            assert len(result["late_confirmation_packet_ids"]) == 1
            assert result["moved_flow_ids"] == []
            assert engine.flows["f"].link == "A"

            # A send after the deadline returns 201 immediately and never 429s,
            # even though the buffer was never released by the callback.
            sent = await engine.send_packet("f", 2)
            assert sent["buffered"] is False
            assert sent["packet"]["status"] == "queued"
            assert sent["packet"]["link"] == "A"

            # Each packet delivered exactly once, in acceptance order.
            await asyncio.sleep(0.3)
            delivery = engine.flows["f"].received
            assert [x["seq"] for x in delivery] == [1, 2]
            assert len({x["packet_id"] for x in delivery}) == 2
        finally:
            await engine.stop()

    asyncio.run(run())


def test_send_after_deadline_flushes_buffer_in_acceptance_order_and_returns_201(client):
    """After the deadline, a new packet first flushes the buffer back to the
    original link in acceptance order, then joins the queue (201, not 429)."""
    create_flow(client, "f", "A")
    send(client, "f", 1)  # ~90ms on A
    body = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 30, "buffer_limit": 2}
    ).json()
    removal_id = body["removal_id"]
    send(client, "f", 2, expect=202)
    send(client, "f", 3, expect=202)
    # Buffer is full: another send during the wait is rejected with 429.
    assert client.post("/api/flows/f/packets", json={"seq": 4}).status_code == 429

    time.sleep(0.05)  # deadline (30ms) has passed
    sent = send(client, "f", 4)  # not 429, not 202: accepted into the link queue
    assert sent["buffered"] is False
    assert sent["removal_id"] is None
    assert sent["packet"]["link"] == "A"

    detail = removal(client, removal_id)
    assert detail["status"] == "timed_out"
    assert detail["flushed_to"] == "A"
    assert detail["moved_flow_ids"] == []
    # Buffered packets are flushed before the new packet joins the queue.
    flushed_seqs = [client.get(f"/api/packets/{pid}").json()["seq"]
                    for pid in detail["flushed_packet_ids"]]
    assert flushed_seqs == [2, 3]

    assert wait_for(client, lambda: len(received(client, "f")) == 4)
    order = received(client, "f")
    assert [item["seq"] for item in order] == [1, 2, 3, 4]
    assert [item["link"] for item in order] == ["A", "A", "A", "A"]
    # Every accepted packet is delivered exactly once.
    assert len({item["packet_id"] for item in order}) == 4


def test_late_confirmation_is_only_recorded_in_removal():
    """Confirmations after the timeout never change the outcome, they only
    show up in the removal's late-confirmation bookkeeping."""

    async def run() -> None:
        engine = DrillEngine(link_a_delivery_ms=20, link_a_ack_ms=20,
                             link_b_delivery_ms=0, link_b_ack_ms=0)
        await engine.start()
        try:
            await engine.create_flow("f", "A")
            await engine.create_flow("g", "B")
            await engine.send_packet("f", 1)
            rm = await engine.create_removal("A", wait_ms=10, buffer_limit=2)
            await asyncio.sleep(0.05)  # timeout fires (~40ms of work on A)
            assert engine.removals[rm["removal_id"]].status == "timed_out"
            await asyncio.sleep(0.1)  # ack for f#1 now arrives
            detail = await engine.get_removal(rm["removal_id"])
            assert detail["status"] == "timed_out"
            assert detail["snapshot_acked"] == 0
            assert detail["late_confirmations_ignored"] == 1
            assert len(detail["late_confirmation_packet_ids"]) == 1
            assert detail["moved_flow_ids"] == []
            assert engine.flows["f"].link == "A"
            # The packet is delivered exactly once, on the member link.
            delivery = engine.flows["f"].received
            assert [x["seq"] for x in delivery] == [1]
            assert delivery[0]["link"] == "A"
        finally:
            await engine.stop()

    asyncio.run(run())


def test_buffered_packets_survive_only_once_through_flush(client):
    """Flushing the buffer must not duplicate packets that were already
    enqueued when the removal finished."""
    create_flow(client, "f", "A")
    send(client, "f", 1)
    body = client.post(
        "/api/removals", json={"member_link": "A", "wait_ms": 5000, "buffer_limit": 5}
    ).json()
    send(client, "f", 2, expect=202)
    assert wait_for(client, lambda: removal(client, body["removal_id"])["status"] == "succeeded")
    send(client, "f", 3, expect=201)
    assert wait_for(client, lambda: len(received(client, "f")) == 3)
    order = received(client, "f")
    assert [item["seq"] for item in order] == [1, 2, 3]
    assert len({item["packet_id"] for item in order}) == 3

